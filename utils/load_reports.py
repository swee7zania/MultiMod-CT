import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import REPORTS

def load_reports(split="train"):
    """
    加载训练或验证集的报告 CSV 文件。
    返回 DataFrame：['id', 'report']
    """
    if split not in REPORTS:
        raise ValueError(f"无效 split：{split}。必须是 'train' 或 'validation'")

    df = pd.read_csv(REPORTS[split])
    return df

if __name__ == "__main__":
    df = load_reports("train")
    print("✅ 报告数据加载成功，前几行如下：")
    print(df.head())