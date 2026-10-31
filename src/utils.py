# utils.py - small helper functions

import os
import config


def create_folders():
    # make the output folders if they are not there
    os.makedirs(config.CHART_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)


def print_heading(text):
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def save_text(text, path):
    with open(path, "w") as f:
        f.write(text)
