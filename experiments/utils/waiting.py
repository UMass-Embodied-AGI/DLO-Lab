import time
import argparse
from datetime import datetime


def arg_parser():
    parser = argparse.ArgumentParser(description="waiting progress")
    parser.add_argument("--seconds", "-s",
                        default=10,
                        type=int,
                        help="Specified the waiting time in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parser()
    wait = arg.seconds
    print(f"Start and wait for {wait}s.")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(wait)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Done.")
