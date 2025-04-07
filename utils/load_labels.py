import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import LABELS

def load_labels(split="train"):
    """
    加载多标签数据，返回 DataFrame。
    """
    if split not in LABELS:
        raise ValueError(f"无效 split：{split}。必须是 'train' 或 'validation'")

    df = pd.read_csv(LABELS[split])
    return df


if __name__ == "__main__":
    df = load_labels("train")
    print("✅ 标签数据加载成功，前几行如下：")
    print(df.head())