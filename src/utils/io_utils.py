import os
import logging
import numpy as np
import scipy.io as sio
from natsort import natsorted


def load_and_combine_mat_files(session_path, key="s"):
    mat_files = [f for f in os.listdir(session_path) if f.endswith(".mat")]
    mat_files = natsorted(mat_files)

    arrays = []
    for f in mat_files:
        file_path = os.path.join(session_path, f)
        try:
            mat = sio.loadmat(file_path)
            if key not in mat:
                logging.debug("MAT file %s does not contain key '%s'", file_path, key)
                continue
            arr = mat[key]
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            arrays.append(arr)
        except Exception as e:
            logging.exception("Failed to read MAT file: %s | %s", file_path, e)

    if not arrays:
        return None

    return np.concatenate(arrays, axis=1)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_metadata_header(writer):
    writer.writerow([
        "filename",
        "class_name",
        "class_id",
        "session_folder",
        "start",
        "end",
        "event_channels",
        "event_channel_scores",
        "window_label",
        "event_ratio",
    ])

def crop_channels(combined, ch_start=20, ch_end=101):
    return combined[ch_start:ch_end, :]