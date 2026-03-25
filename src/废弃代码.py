import os

data_dir = "/home/sente/das_ai_project/data/train/data"  # 改成你的实际路径，比如 /home/sente/xxx/data

# 遍历 data 下的一级目录
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)

    if os.path.isdir(category_path):
        # 统计子文件夹数量
        subfolders = [
            f for f in os.listdir(category_path)
            if os.path.isdir(os.path.join(category_path, f))
        ]

        print(f"{category}: {len(subfolders)} 个子文件夹")