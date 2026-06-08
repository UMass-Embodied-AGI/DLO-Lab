from typing_extensions import Literal

ENDC = '\033[0m'
COLOR_CODE = {
    'blue': '\033[94m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'magenta': '\033[95m',
    'red': '\033[91m',
}


def color_print(text, color: Literal['blue', 'cyan', 'green', 'yellow', 'magenta', 'red'], **kwargs):
    print(f"{COLOR_CODE.get(color, '')}{text}{ENDC}", **kwargs)
