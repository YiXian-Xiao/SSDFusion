import pathlib


def get_latest_checkpoint(dir, name) -> pathlib.Path:
    return pathlib.Path(dir, f'{name}-latest.pth')
