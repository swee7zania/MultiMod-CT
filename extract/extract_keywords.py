import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGE_DIR, REPORTS

# 读取报告数据
report_path = REPORTS["train"]
df = pd.read_csv(report_path)

# 生成图像路径
def get_image_path(volume_name):
    base = volume_name.replace(".nii.gz", "")
    parts = base.split("_")  # ['train', '1', 'a', '1']
    folder_lv1 = f"{parts[0]}_{parts[1]}"
    folder_lv2 = f"{parts[0]}_{parts[1]}_{parts[2]}"
    image_path = os.path.join(IMAGE_DIR, folder_lv1, folder_lv2, volume_name)
    return image_path

# 添加图像路径并过滤掉缺失图像的数据
df["image_path"] = df["VolumeName"].apply(get_image_path)
df["image_exists"] = df["image_path"].apply(os.path.exists)
df_filtered = df[df["image_exists"]].drop(columns=["image_exists"])

# 填空值并拼接 full_report
df_filtered["Findings_EN"] = df_filtered["Findings_EN"].fillna("")
df_filtered["Impressions_EN"] = df_filtered["Impressions_EN"].fillna("")
df_filtered["full_report"] = df_filtered["Findings_EN"] + " " + df_filtered["Impressions_EN"]

# Impression only 版本
df_filtered["impression_only"] = df_filtered["Impressions_EN"]

# ----------------------
# 🔍 添加结构关键词提取列
# ----------------------

# 关键词词典
organs = ["lung", "heart", "pleura", "mediastinum", "lymph", "trachea", "bronchi"]
abnormalities = ["nodule", "opacity", "effusion", "calcification", "atelectasis", "emphysema"]

def extract_keywords(text, keywords):
    if pd.isna(text): return []
    text = text.lower()
    return list({kw for kw in keywords if kw in text})

def classify_impression(text):
    if pd.isna(text): return "unknown"
    text = text.lower()
    if any(word in text for word in ["infection", "pneumonia", "abscess"]):
        return "infection"
    elif any(word in text for word in ["tumor", "malignant", "neoplasm"]):
        return "tumor"
    elif any(word in text for word in ["inflammation", "fibrosis"]):
        return "inflammation"
    elif any(word in text for word in ["normal", "unremarkable"]):
        return "normal"
    else:
        return "other"

# 应用提取逻辑
df_filtered["Mentioned_Organs"] = df_filtered["Findings_EN"].apply(lambda x: extract_keywords(x, organs))
df_filtered["Abnormality_Keywords"] = df_filtered["Findings_EN"].apply(lambda x: extract_keywords(x, abnormalities))
df_filtered["Impression_Category"] = df_filtered["Impressions_EN"].apply(classify_impression)

# 构建 keyword prompt 文本
def build_keyword_prompt(row):
    parts = []
    if row["Mentioned_Organs"]:
        parts.append("Organs: " + ", ".join(row["Mentioned_Organs"]))
    if row["Abnormality_Keywords"]:
        parts.append("Findings: " + ", ".join(row["Abnormality_Keywords"]))
    if row["Impression_Category"]:
        parts.append("Diagnosis: " + row["Impression_Category"])
    return " | ".join(parts)

df_filtered["keyword_prompt"] = df_filtered.apply(build_keyword_prompt, axis=1)

# ----------------------
# 🎯 最终导出字段
# ----------------------
df_final = df_filtered[[
    "VolumeName", "image_path", "full_report", "impression_only", "keyword_prompt"
]]

# 保存最终文件
df_final.to_csv("train_reports_with_prompt_styles.csv", index=False)

# 输出信息
print(f"原始记录数: {len(df)}")
print(f"保留有图像的记录数: {len(df_final)}")
print("示例 prompt:\n", df_final["keyword_prompt"].iloc[0])
