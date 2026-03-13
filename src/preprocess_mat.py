import os
import csv
import argparse
import numpy as np
import scipy.io as sio
from natsort import natsorted
import codecs
import re


# -------------------------------------------------
# 解析 label txt
# -------------------------------------------------
def parse_label_txt(label_path):

    info = {
        "class": None,
        "channel": None,
        "intensity": None,
        "env": ""
    }

    try:
        with codecs.open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                if '=' in line:
                    parts = line.split('=', 1)
                    key = parts[0].strip().lower()
                    val = parts[1].strip()

                else:
                    parts = line.split()

                    if len(parts) >= 2 and parts[0].strip().endswith("类别"):
                        key = "class"
                        val = parts[-1]
                    else:
                        key = "env"
                        val = line

                if "data_path" in key or "类别" in key or "class" in key:

                    info["class"] = val

                elif "channel" in key:

                    m = re.search(r"(\d+)", val)

                    if m:
                        info["channel"] = int(m.group(1))

                elif "intensity" in key or "强度" in key:

                    m = re.search(r"(\d+)", val)

                    if m:
                        info["intensity"] = int(m.group(1))

                else:

                    if info["env"]:
                        info["env"] += " | " + val
                    else:
                        info["env"] = val

    except Exception:
        pass

    return info


# -------------------------------------------------
# 找 label
# -------------------------------------------------
def find_label_for_event_and_parse(label_root, class_name, session_folder):

    candidate_dir = os.path.join(label_root, class_name)

    if not os.path.isdir(candidate_dir):
        return None, None

    for fn in os.listdir(candidate_dir):

        name, ext = os.path.splitext(fn)

        s1 = session_folder.replace("_", "").replace("-", "")
        s2 = name.replace("_", "").replace("-", "")

        if s1 in s2 or s2 in s1:

            full = os.path.join(candidate_dir, fn)

            if ext.lower() == ".txt":

                parsed = parse_label_txt(full)

                return full, parsed

    return None, None


# -------------------------------------------------
# 读取 mat
# -------------------------------------------------
def load_and_combine_mat_files(session_path):

    mat_files = [f for f in os.listdir(session_path) if f.endswith(".mat")]

    mat_files = natsorted(mat_files)

    arrays = []

    for f in mat_files:

        file_path = os.path.join(session_path, f)

        try:

            mat = sio.loadmat(file_path)

            if "s" not in mat:
                continue

            arr = mat["s"]

            arrays.append(arr)

        except Exception as e:
            print("读取失败:", file_path, e)

    if not arrays:
        return None

    combined = np.concatenate(arrays, axis=1)

    return combined


# -------------------------------------------------
# 保存窗口
# -------------------------------------------------
def save_windows_and_metadata(
        combined_arr,
        out_dir,
        class_name,
        session_folder,
        window_len,
        hop,
        label_path,
        label_parsed,
        meta_writer
):

    ch, total = combined_arr.shape

    os.makedirs(out_dir, exist_ok=True)

    idx = 0

    for start in range(0, total - window_len + 1, hop):

        win = combined_arr[:, start:start + window_len]

        save_arr = win.T.astype(np.float32)

        fname = f"{class_name}__{session_folder}__{idx:06d}.npy"

        path = os.path.join(out_dir, fname)

        np.save(path, save_arr)

        lp = label_path if label_path else ""

        chn = label_parsed.get("channel") if label_parsed else ""

        inten = label_parsed.get("intensity") if label_parsed else ""

        env = label_parsed.get("env") if label_parsed else ""

        meta_writer.writerow([
            fname,
            class_name,
            session_folder,
            start,
            start + window_len,
            lp,
            chn,
            inten,
            env
        ])

        idx += 1

    return idx


# -------------------------------------------------
# 主函数
# -------------------------------------------------
def main(args):

    data_root = args.data_root
    label_root = args.label_root
    out_root = args.out_root

    window_len = args.window_len
    hop = args.hop

    os.makedirs(out_root, exist_ok=True)

    metadata_path = os.path.join(out_root, "metadata.csv")

    with open(metadata_path, "w", newline='', encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "class",
            "session_folder",
            "start",
            "end",
            "label_path",
            "label_channel",
            "label_intensity",
            "label_env"
        ])

        for class_name in os.listdir(data_root):

            class_dir = os.path.join(data_root, class_name)

            if not os.path.isdir(class_dir):
                continue

            print("处理类别:", class_name)

            out_class_dir = os.path.join(out_root, class_name)

            for session_folder in os.listdir(class_dir):

                session_path = os.path.join(class_dir, session_folder)

                if not os.path.isdir(session_path):
                    continue

                print("  session:", session_folder)

                combined = load_and_combine_mat_files(session_path)

                if combined is None:
                    print("  无有效数据")
                    continue

                label_path, label_parsed = find_label_for_event_and_parse(
                    label_root,
                    class_name,
                    session_folder
                )

                save_windows_and_metadata(
                    combined,
                    out_class_dir,
                    class_name,
                    session_folder,
                    window_len,
                    hop,
                    label_path,
                    label_parsed,
                    writer
                )

    print("预处理完成")
    print("metadata:", metadata_path)


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--window_len", type=int, default=100)
    parser.add_argument("--hop", type=int, default=50)

    args = parser.parse_args()

    main(args)