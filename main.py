import argparse
from pathlib import Path

from handtracker import (
    CameraSource,
    Config,
    FingerStateClassifier,
    FpsRenderer,
    GestureEngine,
    GestureHudRenderer,
    GestureRecorder,
    HandLabelRenderer,
    HandTrackerApp,
    LandmarkRenderer,
    MediaPipeConfig,
    MediaPipeHandDetector,
    PinchRenderer,
    SkeletonRenderer,
    make_emitter,
    make_key_capture,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Жесты рук -> нажатия клавиш")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="не нажимать клавиши, только печатать в консоль")
    args = parser.parse_args()

    config = Config.load(args.config)
    settings = config.settings
    print(f"[config] {config.path}")

    detector = MediaPipeHandDetector(MediaPipeConfig(
        num_hands=2, swap_handedness=settings.swap_handedness))
    classifier = FingerStateClassifier(settings.thresholds)
    engine = GestureEngine(config, classifier, make_emitter(args.dry_run or settings.dry_run))

    capture = make_key_capture()
    if capture is None:
        print("[keys] pynput недоступен: при записи жеста клавиша ловится только в окне OpenCV")
    recorder = GestureRecorder(config, classifier, engine, capture)

    points = LandmarkRenderer(show_ids=True, override=engine.color_of)
    pinch = PinchRenderer()
    pinch.enabled = False

    app = HandTrackerApp(
        source=CameraSource(args.camera),
        detector=detector,
        renderers=[
            SkeletonRenderer(override=engine.color_of),  # жест с цветом красит всю кисть
            points,
            HandLabelRenderer(),
            pinch,
            GestureHudRenderer(engine, recorder),
            FpsRenderer(),
        ],
        processors=[recorder, engine],
        mirror=settings.mirror,
    )

    def toggle_mirror() -> None:
        app.toggle_mirror()
        settings.mirror = app.mirror
        config.save()

    def toggle_swap() -> None:
        detector.toggle_swap()
        settings.swap_handedness = detector.config.swap_handedness
        config.save()

    app.bind("n", "ids", points.toggle_ids)
    app.bind("p", "pinch", pinch.toggle)
    app.bind("m", "mirror", toggle_mirror)
    app.bind("h", "swap L/R", toggle_swap)
    app.bind("k", "keys on/off", engine.toggle)
    app.bind("b", "bind gesture", recorder.start)
    app.run()


if __name__ == "__main__":
    main()
