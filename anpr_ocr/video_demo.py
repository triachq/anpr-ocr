"""CLI demo: runs ANPR-OCR on a video file and writes an annotated output."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import cast

from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from anpr_ocr import ALPR, PlateLogger, VehicleRecord
from anpr_ocr.alpr import SUPPORTED_VIDEO_EXTS

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "video_results"

# Common Indian & international plate syntax patterns for disambiguation
PLATE_PATTERNS = [
    "LLDDLLDDDD",  # 10-char: MH12DE1433, HR36AE7971
    "LLDDLDDDD",   # 9-char:  TN45Q3566
    "LLDDDDDD",    # 8-char:  LA020749
    "LLDDDDD",     # 7-char:  IN03044
    "DLLDDDD",     # 7-char:  5AU5341
    "LLDDLLL",     # 7-char:  LB02APF
    "LLLDDD",      # 6-char:  IZX842
]

# -- ANSI colors (enable on Windows) -----------------------------------------
os.system("")  # enable VT100 on Windows

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"


def _bar(ratio: float, width: int = 20) -> str:
    """Render a confidence gauge."""
    filled = round(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ANPR-OCR on video file(s) and produce annotated output video(s).",
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a video file or directory (default: scans data/videos/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file or directory (default: artifacts/video_results/<name>_anpr.mp4).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame, 3 = every 3rd). Default: 1.",
    )
    parser.add_argument(
        "--detector",
        default="yolo-v9-s-608-license-plate-end2end",
        help="Detector model name (default: yolo-v9-s-608-license-plate-end2end).",
    )
    parser.add_argument(
        "--ocr",
        default="cct-s-v2-global-model",
        help="OCR model name (default: cct-s-v2-global-model).",
    )
    parser.add_argument(
        "--codec",
        default=None,
        help="FourCC codec for the output video (e.g. mp4v, XVID). Auto-detected by default.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="Detector confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--syntax",
        nargs="*",
        default=None,
        help="Syntax mask pattern(s) for disambiguation (e.g. LLDDLLDDDD).",
    )
    parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Apply CLAHE contrast enhancement before OCR.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="Minimum number of recognized characters to display a plate (default: 4).",
    )
    parser.add_argument(
        "--play",
        "--live",
        dest="play",
        action="store_true",
        help="Play video in a live real-time GUI window with detection overlays (Press Q to quit).",
    )
    parser.add_argument(
        "--directml",
        action="store_true",
        default=False,
        help="Enable DirectML GPU acceleration (Intel Iris Xe / AMD / NVIDIA).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Custom CSV file path for logging peak license plate records (default: auto in output dir).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Disable automatic CSV plate logging.",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        default=False,
        help="Save high-resolution plate crop snapshot images for each finalized vehicle.",
    )
    parser.add_argument(
        "--min-log-conf",
        type=float,
        default=0.50,
        help="Minimum confidence threshold for logging a unique vehicle (default: 0.50).",
    )
    return parser


def _make_progress_callback(total_frames: int):
    """Return a callback that prints a live progress bar."""
    last_pct = [-1]  # mutable to capture in closure
    start = time.perf_counter()

    def _callback(current: int, total: int) -> None:
        pct = int(current / max(total, 1) * 100)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        elapsed = time.perf_counter() - start
        fps = current / elapsed if elapsed > 0 else 0
        bar_w = 30
        filled = round(pct / 100 * bar_w)
        bar = "#" * filled + "-" * (bar_w - filled)
        eta = (total - current) / fps if fps > 0 else 0
        sys.stdout.write(
            f"\r  {CYAN}[{bar}]{RESET} {pct:3d}%  "
            f"{DIM}{current}/{total} frames  "
            f"{fps:.1f} fps  "
            f"ETA {eta:.0f}s{RESET}  "
        )
        sys.stdout.flush()

    return _callback


def _play_video_live(
    video_path: Path,
    alpr: ALPR,
    frame_skip: int = 2,
    min_chars: int = 4,
    logger: PlateLogger | None = None,
) -> None:
    """Play video in an interactive real-time GUI window with detection overlays."""
    import cv2
    import statistics

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"{RED}Error: Failed to open video for playback: {video_path}{RESET}")
        return

    win_name = f"ANPR Real-Time - {video_path.name} (Press Q or ESC to exit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = max(1, int(1000 / src_fps))

    from anpr_ocr.utils import PlateTracker

    tracker = PlateTracker(max_unseen_frames=frame_skip * 5, window_size=5)
    active_plates: list[tuple] = []
    frame_idx = 0
    t_start = time.perf_counter()

    print(f"  {GREEN}[LIVE]{RESET} Playing {video_path.name} in GUI window...")
    print(f"  {DIM}Controls: Press 'Q' or 'ESC' to exit, [SPACE] to pause, [N] to step.{RESET}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.perf_counter() - t_start
        fps_live = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        # Deep inference on every Nth frame
        if frame_idx % frame_skip == 0:
            res = alpr.predict(frame)
            frame_dets = []
            for r in res:
                if not r.ocr or not r.ocr.text or len(r.ocr.text.strip()) < min_chars:
                    continue
                conf = (
                    statistics.mean(r.ocr.confidence)
                    if isinstance(r.ocr.confidence, list)
                    else (r.ocr.confidence or 0.0)
                )
                if conf < 0.35:
                    continue
                frame_dets.append((r.detection.bounding_box, r.ocr.text.strip(), conf))

            tracked = tracker.update(frame_dets, frame_idx)
            active_plates = [(box, txt, conf) for box, txt, conf, _ in tracked]

            if logger is not None:
                for box, txt, conf in active_plates:
                    logger.observe(
                        plate_text=txt,
                        confidence=conf,
                        bounding_box=box,
                        frame_idx=frame_idx,
                        frame_bgr=frame,
                        fps=src_fps,
                    )

        # Render annotations
        display = frame.copy()
        for b, txt, c in active_plates:
            cv2.rectangle(display, (b.x1, b.y1), (b.x2, b.y2), (36, 255, 12), 2)
            lbl = f"{txt} {c * 100:.0f}%"
            cv2.putText(
                display, lbl, (b.x1, max(b.y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                display, lbl, (b.x1, max(b.y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )

        # HUD Banner
        veh_count = len(logger.finalize()) if logger is not None else 0
        hud = f"Live FPS: {fps_live:.1f} | Frame: {frame_idx} | Vehicles: {veh_count} | [SPACE] Pause | [Q] Quit"
        cv2.putText(display, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord(" "), ord("p"), ord("P")):
            paused = True
            while paused:
                pause_display = display.copy()
                pause_hud = f"[PAUSED] Frame: {frame_idx} | Vehicles: {veh_count} | [SPACE] Resume | [N] Step Next | [Q] Quit"
                cv2.putText(pause_display, pause_hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(pause_display, pause_hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow(win_name, pause_display)
                p_key = cv2.waitKey(30) & 0xFF
                if p_key in (ord("q"), ord("Q"), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"  {GREEN}[OK] Live playback closed.{RESET}")
                    if logger is not None:
                        print()
                        print(logger.summary_table())
                        if logger.output_csv:
                            csv_p = logger.export_csv()
                            if csv_p:
                                print(f"  {GREEN}[LOG]{RESET} Finalized peak plate log saved to: {csv_p}")
                        if logger.snapshots_dir:
                            snaps = logger.save_snapshots()
                            if snaps:
                                print(f"  {GREEN}[LOG]{RESET} Saved {len(snaps)} vehicle plate snapshot(s) to: {logger.snapshots_dir}")
                        print()
                    return
                elif p_key in (ord(" "), ord("p"), ord("P")):
                    paused = False
                    break
                elif p_key in (ord("n"), ord("N")):
                    # Step forward one frame
                    break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"  {GREEN}[OK] Live playback finished.{RESET}")

    if logger is not None:
        print()
        print(logger.summary_table())
        if logger.output_csv:
            csv_p = logger.export_csv()
            if csv_p:
                print(f"  {GREEN}[LOG]{RESET} Finalized peak plate log saved to: {csv_p}")
        if logger.snapshots_dir:
            snaps = logger.save_snapshots()
            if snaps:
                print(f"  {GREEN}[LOG]{RESET} Saved {len(snaps)} vehicle plate snapshot(s) to: {logger.snapshots_dir}")
        print()


def main() -> int:
    args = _build_parser().parse_args()

    # -- Resolve video files -------------------------------------------------
    if args.video is None:
        target = DEFAULT_VIDEOS_DIR
    else:
        target = args.video.resolve()

    video_files: list[Path] = []
    if target.is_dir():
        video_files = sorted(
            p for p in target.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        if not video_files:
            print(f"{RED}Error: No supported video files found in: {target}{RESET}")
            print(f"  Please place video files ({', '.join(sorted(SUPPORTED_VIDEO_EXTS))}) there,")
            print("  or pass a video path: uv run video-demo path/to/video.mp4")
            return 1
    elif target.is_file():
        ext = target.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            print(
                f"{RED}Error: Unsupported video format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}{RESET}"
            )
            return 1
        video_files = [target]
    else:
        print(f"{RED}Error: Video path not found: {target}{RESET}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    syntax = args.syntax if args.syntax else PLATE_PATTERNS

    # -- Banner --------------------------------------------------------------
    print()
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  ANPR-OCR Video Demo -- Automatic Number Plate Recognition{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print()
    print(f"  {DIM}Target:{RESET}      {target}")
    print(f"  {DIM}Videos found:{RESET}{len(video_files)} file(s)")
    for v in video_files:
        print(f"    - {v.name}")
    print(f"  {DIM}Detector:{RESET}   {args.detector}")
    print(f"  {DIM}OCR:{RESET}        {args.ocr}")
    print(f"  {DIM}Frame skip:{RESET} {args.frame_skip}")
    if args.codec:
        print(f"  {DIM}Codec:{RESET}      {args.codec}")
    print()

    # -- Configure Execution Providers (DirectML GPU or CPU) ---------------
    providers = None
    if args.directml:
        import onnxruntime as ort
        if "DmlExecutionProvider" in ort.get_available_providers():
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            print(f"  {GREEN}[GPU] DirectML hardware acceleration ENABLED (Intel Iris Xe / GPU){RESET}")
        else:
            print(f"  {YELLOW}[WARN] DirectML provider not available, falling back to CPU{RESET}")

    # -- Load models ---------------------------------------------------------
    print(f"  {YELLOW}Loading models...{RESET}", end="", flush=True)
    t0 = time.perf_counter()
    alpr = ALPR(
        detector_model=cast(PlateDetectorModel, args.detector),
        ocr_model=cast(OcrModel, args.ocr),
        detector_conf_thresh=args.conf_thresh,
        enhance_contrast=args.enhance_contrast,
        detector_providers=providers,
        ocr_providers=providers,
        syntax_pattern=syntax,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"\r  {GREEN}[OK] Models loaded in {load_ms:.0f} ms{RESET}            ")
    print()

    # -- Process videos ------------------------------------------------------
    import cv2
    import statistics

    all_results = []
    total_pipeline_time = 0.0

    for idx, video_path in enumerate(video_files, 1):
        # File Export / Output Paths
        if args.output and len(video_files) == 1 and args.output.suffix:
            output_path = args.output.resolve()
        elif args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            output_path = args.output.resolve() / f"{video_path.stem}_anpr{video_path.suffix}"
        else:
            output_path = OUTPUT_DIR / f"{video_path.stem}_anpr{video_path.suffix}"

        # Resolve CSV log path and snapshots directory
        if not args.no_csv:
            if args.csv and len(video_files) == 1:
                csv_path = args.csv.resolve()
            else:
                csv_path = output_path.parent / f"{video_path.stem}_plates.csv"
        else:
            csv_path = None

        snapshots_dir = (output_path.parent / f"{video_path.stem}_snapshots") if args.snapshots else None

        logger = PlateLogger(
            output_csv=csv_path,
            snapshots_dir=snapshots_dir,
            min_conf=args.min_log_conf,
            min_chars=args.min_chars,
        )

        # Interactive Live Playback Mode
        if args.play:
            _play_video_live(
                video_path=video_path,
                alpr=alpr,
                frame_skip=args.frame_skip,
                min_chars=args.min_chars,
                logger=logger,
            )
            continue

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"  {BOLD}[{idx}/{len(video_files)}] Processing: {video_path.name}{RESET}")
        print(f"      {DIM}Resolution:{RESET} {width}x{height} @ {src_fps:.1f} fps, {total_frames} frames")
        print(f"      {DIM}Saving to:{RESET}  {output_path.name}")

        progress = _make_progress_callback(total_frames)
        t_start = time.perf_counter()
        result = alpr.draw_predictions_video(
            source=video_path,
            output_path=output_path,
            frame_skip=args.frame_skip,
            codec=args.codec,
            min_chars=args.min_chars,
            progress_callback=progress,
            logger=logger,
        )
        elapsed = time.perf_counter() - t_start
        total_pipeline_time += elapsed

        # Clear progress line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        print(
            f"      {GREEN}[OK]{RESET} {result.processed_frames}/{result.total_frames} frames | "
            f"{result.total_plates_detected} plates detected | "
            f"{result.fps_processing:.1f} fps ({result.processing_time_seconds:.1f}s)"
        )
        print()

        # Display Finalized Peak Vehicle Log
        print(logger.summary_table())
        if csv_path:
            saved_csv = logger.export_csv()
            if saved_csv:
                print(f"      {GREEN}[LOG]{RESET} Finalized peak plate log saved to: {saved_csv.name}")
        if snapshots_dir:
            snaps = logger.save_snapshots()
            if snaps:
                print(f"      {GREEN}[LOG]{RESET} Saved {len(snaps)} peak plate snapshot(s) to: {snapshots_dir.name}/")
        print()
        all_results.append(result)

    # -- Summary -------------------------------------------------------------
    total_frames_all = sum(r.total_frames for r in all_results)
    proc_frames_all = sum(r.processed_frames for r in all_results)
    plates_all = sum(r.total_plates_detected for r in all_results)
    unique_vehicles_all = sum(len(r.vehicle_records) for r in all_results if r.vehicle_records is not None)
    overall_fps = proc_frames_all / total_pipeline_time if total_pipeline_time > 0 else 0

    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  Overall Summary{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"  {BOLD}Videos processed:{RESET}       {len(all_results)}")
    print(f"  {BOLD}Total frames:{RESET}           {total_frames_all}")
    print(f"  {BOLD}Processed frames:{RESET}       {proc_frames_all}")
    print(f"  {BOLD}Total plates detected:{RESET}  {plates_all}")
    print(f"  {BOLD}Unique vehicles logged:{RESET} {unique_vehicles_all} (peak confidence deduplicated)")
    print(f"  {BOLD}Total processing time:{RESET}  {total_pipeline_time:.2f}s")
    print(f"  {BOLD}Effective throughput:{RESET}   {overall_fps:.1f} fps")
    print()
    out_dir_display = args.output if (args.output and args.output.is_dir()) else OUTPUT_DIR
    print(f"  {GREEN}[OK] Annotated videos saved to:{RESET} {out_dir_display}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
