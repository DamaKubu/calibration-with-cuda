import argparse


def backend_name(cv2, backend: int) -> str:
    if backend == cv2.CAP_DSHOW:
        return "dshow"
    if backend == cv2.CAP_MSMF:
        return "msmf"
    if backend == cv2.CAP_ANY:
        return "any"
    return str(backend)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe camera backends / modes and report actual frame sizes.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    try:
        import cv2  # type: ignore
    except Exception as e:
        print("cv2 import failed:", repr(e))
        print("If you want to install it just for debugging: pip install opencv-python")
        return 2

    cam = args.camera

    def test(backend: int, fourcc: str | None = None, w: int | None = None, h: int | None = None):
        cap = cv2.VideoCapture(cam, backend)
        if not cap.isOpened():
            return None

        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if w is not None and h is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))

        ok, frame = cap.read()
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_hw = None if not ok or frame is None else (frame.shape[0], frame.shape[1])

        cap.release()

        return {
            "backend": backend,
            "fourcc": fourcc,
            "req": (w, h),
            "got": (got_w, got_h),
            "frame": frame_hw,
        }

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    sizes = [None, (1280, 720), (1280, 960), (1920, 1080), (640, 480), (800, 600)]

    print("cv2", cv2.__version__)
    print("camera", cam)

    results = []
    for backend in backends:
        for fourcc in [None, "MJPG"]:
            for s in sizes:
                if s is None:
                    r = test(backend, fourcc=fourcc)
                else:
                    r = test(backend, fourcc=fourcc, w=s[0], h=s[1])
                if r is not None:
                    results.append(r)

    if not results:
        print("No backends opened the camera.")
        return 1

    # Summarize by backend+fourcc: what unique actual frame sizes appear?
    by_bf = {}
    for r in results:
        key = (r["backend"], r["fourcc"])
        by_bf.setdefault(key, []).append(r)

    print("\nSummary (unique actual frames per backend):")
    for (backend, fourcc), rows in by_bf.items():
        name = backend_name(cv2, backend)
        # Collect unique frame shapes and which requests produced them.
        frame_to_reqs = {}
        for r in rows:
            frame_to_reqs.setdefault(r["frame"], []).append(r["req"])

        print(f"- backend={name} fourcc={fourcc}:")
        for frame_hw, reqs in frame_to_reqs.items():
            sample_reqs = ", ".join([str(x) for x in reqs[:4]])
            more = "" if len(reqs) <= 4 else f" (+{len(reqs) - 4} more)"
            # Also show the last reported CAP_PROP width/height for this frame size
            got = None
            for r in rows:
                if r["frame"] == frame_hw:
                    got = r["got"]
                    break
            print(f"  frame={frame_hw} cap_prop={got} from req {sample_reqs}{more}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
