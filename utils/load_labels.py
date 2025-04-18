import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    # 去除 ID 列，只保留标签部分
    label_cols = df.columns.tolist()[1:]
    label_matrix = df[label_cols]

    # 1️⃣ 每个标签的出现频次
    label_counts = label_matrix.sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="Blues_d")
    plt.title("Number of occurrences per label (training set)")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 2️⃣ 每个样本有多少个标签（多标签数量分布）
    label_per_sample = label_matrix.sum(axis=1)

    plt.figure(figsize=(8, 4))
    sns.histplot(label_per_sample, bins=range(1, label_per_sample.max() + 2), discrete=True)
    plt.title("The number of labels contained in each sample")
    plt.xlabel("Number of labels")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.show()
