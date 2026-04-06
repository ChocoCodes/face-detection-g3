from _runner import run_base


if __name__ == "__main__":
    import sys

    raise SystemExit(run_base("face_detect.py", sys.argv[1:]))

