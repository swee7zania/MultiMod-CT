import os
import sys
import json
import random
import pandas as pd

# 添加 config 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import METADATA

# 设置参数
N_NEGATIVE = 3
SEED = 42
random.seed(SEED)

# 读取主文件（full_report 格式）
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

# 生成 JSONL 文件（使用 full_report）
output_path = "ctrate_mcqa_full_report.jsonl"
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, row in df.iterrows():
        correct_text = row["full_report"]
        image_path = row["image_path"]
        metadata_prompt = row["metadata_prompt"]

        # 获取负样本（避开自己）
        candidates = df[df["image_path"] != image_path]
        negatives = candidates.sample(n=N_NEGATIVE, random_state=SEED + idx)["full_report"].tolist()

        # 构造 options
        options = [{"text": correct_text, "label": 1}] + [{"text": txt, "label": 0} for txt in negatives]
        random.shuffle(options)

        # 写入 JSONL
        record = {
            "image_path": image_path,
            "metadata_prompt": metadata_prompt,
            "options": options
        }

        fout.write(json.dumps(record) + "\n")

print(f"✅ 多模态选择题 JSONL 文件已生成，共包含样本数: {len(df)}")
print(f"📁 文件路径: {output_path}")
