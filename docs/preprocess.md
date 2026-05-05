# DAS Dataset Preprocessing Output Format

本项目将原始 **DAS `.mat` 数据**预处理为 **可用于深度学习训练的数据集格式**。
预处理完成后会生成：

* **`.npy` 数据文件（模型输入）**
* **`metadata.csv`（样本索引与标签信息）**

---

# 1 Output Directory Structure

预处理后的数据存放在：

```
out_root/
```

目录结构如下：

```
processed_data/
│
├── metadata.csv
│
├── pawang/
│   ├── pawang__2021-10-22_14_37_20__000000.npy
│   ├── pawang__2021-10-22_14_37_20__000001.npy
│   ├── pawang__2021-10-22_14_37_20__000002.npy
│   └── ...
│
└── zuanwang/
    ├── zuanwang__2021-10-22_14_41_03__000000.npy
    ├── zuanwang__2021-10-22_14_41_03__000001.npy
    └── ...
```

每个 `.npy` 文件对应 **一个时间窗口样本**。

---

# 2 NPY File Format

每个 `.npy` 文件保存一个 **DAS 时空窗口数据**。

数据来源：

```
多个 .mat 文件
        ↓
拼接时间轴
        ↓
滑动窗口切片
        ↓
保存为 .npy
```

---

## Shape

```
(window_len, channels)
```

例如：

```
(100, 118)
```

说明：

| 维度         | 含义      |
| ---------- | ------- |
| window_len | 时间窗口长度  |
| channels   | DAS 通道数 |

---

## 数据类型

```
float32
```

---

## 示例读取

```python
import numpy as np

data = np.load("pawang__2021-10-22_14_37_20__000000.npy")

print(data.shape)
```

输出示例：

```
(100, 118)
```

---

# 3 Sliding Window Parameters

窗口参数在预处理脚本中设置：

```
window_len = 100
hop = 50
```

含义：

| 参数         | 含义     |
| ---------- | ------ |
| window_len | 时间窗口长度 |
| hop        | 滑动步长   |

窗口示意：

```
time →
|----100----|
     |----100----|
          |----100----|
```

---

# 4 metadata.csv Format

`metadata.csv` 是 **整个数据集的索引文件**。

每一行对应 **一个 `.npy` 样本**。

---

## CSV Structure

```
filename,class,session_folder,start,end,label_channel,label_intensity,label_env
```

字段说明：

| 字段              | 含义         |
| --------------- | ---------- |
| filename        | npy 文件名    |
| class           | 类别         |
| session_folder  | 事件采集时间     |
| start           | 窗口起始位置     |
| end             | 窗口结束位置     |
| label_channel   | 事件通道       |
| label_intensity | 振动强度       |
| label_env       | 采集环境       |

---

## 示例

```
filename,class,session_folder,start,end,label_channel,label_intensity,label_env
pawang__2021-10-22_14_37_20__000000.npy,pawang,2021-10-22_14_37_20,0,100,70,2,立柱
pawang__2021-10-22_14_37_20__000001.npy,pawang,2021-10-22_14_37_20,50,150,70,2,立柱
```

---

# 5 Data Generation Pipeline

完整数据处理流程：

```
Raw DAS data (.mat)
        │
        │ load_and_combine_mat_files
        ▼
Combined signal matrix
(channels × time)
        │
        │ sliding window
        ▼
(window_len × channels)
        │
        │ save
        ▼
.npy samples
        │
        │ metadata recording
        ▼
metadata.csv
```

---

# 6 Dataset Usage in Training

训练时的数据读取流程：

```
metadata.csv
      │
      │ locate npy file
      ▼
load npy
      │
      ▼
PyTorch Dataset
      │
      ▼
Neural Network
```

---

# 7 Example Dataset Loader

示例读取：

```python
import pandas as pd
import numpy as np

meta = pd.read_csv("metadata.csv")

row = meta.iloc[0]

path = "processed_data/" + row["class"] + "/" + row["filename"]

data = np.load(path)

print(data.shape)
```

---

# 8 Key Advantages of This Format

这种数据格式适合机器学习训练，因为：

* `.npy` **加载速度远快于 `.mat`**
* `metadata.csv` **统一管理标签**
* 便于 **PyTorch Dataset 构建**
* 支持 **大规模 DAS 数据**

---

# 9 Notes

1️⃣ 每个 `.npy` 文件只包含 **一个时间窗口样本**

2️⃣ 所有标签信息都在：

```
metadata.csv
```

3️⃣ 原始 `.mat` 数据不再参与训练