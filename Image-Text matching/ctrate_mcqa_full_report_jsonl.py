import os
import sys
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加 config 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import METADATA

# 设置参数
N_NEGATIVE = 3
SEED = 42
random.seed(SEED)

# 读取主文件（包含 image_path, full_report）
df_prompt = pd.read_csv("../extract/cleaned_train_reports_with_full_text.csv")

# 读取 metadata
metadata_df = pd.read_csv(METADATA["train"])

# 清理 VolumeName 并合并
metadata_df["VolumeName"] = metadata_df["VolumeName"].str.strip()
df_prompt["VolumeName"] = df_prompt["VolumeName"].str.strip()
df = df_prompt.merge(metadata_df, on="VolumeName", how="inner")

# 构建 metadata prompt 字段
def format_metadata_prompt(row):
    age = row.get("PatientAge", "")
    sex = row.get("PatientSex", "")
    slice_thickness = row.get("SliceThickness", "")
    return f"Age: {age}, Sex: {sex}, SliceThickness: {slice_thickness}mm"

df["metadata_prompt"] = df.apply(format_metadata_prompt, axis=1)

# 按 80/20 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
splits = [("train", train_df), ("val", val_df)]

# 生成两个 JSONL 文件
for split_name, split_df in splits:
    output_path = f"ctrate_mcqa_full_report_{split_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, row in split_df.iterrows():
            correct_text = row["full_report"]
            image_path = row["image_path"]
            metadata_prompt = row["metadata_prompt"]

            # 获取负样本（避开当前样本）
            candidates = split_df[split_df["image_path"] != image_path]
            if len(candidates) < N_NEGATIVE:
                continue  # 跳过太少样本的情况

            negatives = candidates.sample(n=N_NEGATIVE, random_state=SEED + idx)["full_report"].tolist()

            # 构造选项（打乱）
            options = [{"text": correct_text, "label": 1}] + [{"text": txt, "label": 0} for txt in negatives]
            random.shuffle(options)

            # 记录结构
            record = {
                "image_path": image_path,
                "metadata_prompt": metadata_prompt,
                "options": options
            }
            fout.write(json.dumps(record) + "\n")

    print(f"✅ {split_name.upper()} JSONL 已生成 | 样本数: {len(split_df)}")
    print(f"📁 输出路径: {output_path}")
