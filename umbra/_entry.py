"""Runs INSIDE Blender:  blender -b --factory-startup --python umbra/_entry.py -- '<json>'"""
import sys, os, json, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    args = json.loads(sys.argv[sys.argv.index("--") + 1])
    from umbra import render
    render.run(args)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        print("UMBRA_ERROR_BEGIN")
        traceback.print_exc(file=sys.stdout)
        print("UMBRA_ERROR_END")
        sys.stdout.flush()
        os._exit(3)
