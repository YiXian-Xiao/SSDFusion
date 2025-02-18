import os


def find_available_increasing_name(basepath, file_prefix, start=0, end=500, step=1):
    for i in range(start, end, step):
        path = os.path.join(basepath, f'{file_prefix}{i}')
        if not os.path.exists(path):
            return f'{file_prefix}{i}'
