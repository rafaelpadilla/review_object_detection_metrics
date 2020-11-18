import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, '..', 'src')
add_path(libPath)
libPath = os.path.join(currentPath, '..', 'src', 'utils')
add_path(libPath)
