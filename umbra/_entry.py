"""Runs INSIDE Blender:  blender -b --factory-startup --python umbra/_entry.py -- '<json>'"""
import sys, os, json, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    args = json.loads(sys.argv[sys.argv.index("--") + 1])
    from umbra import render
    render.run(args)
    if str(args.get("mode", "")).startswith("live"):     # nothing to save: skip Blender's shutdown (it is slow, and
        sys.stdout.flush()                               # has crashed there with dozens of processes exiting at once)
        os._exit(0)


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
