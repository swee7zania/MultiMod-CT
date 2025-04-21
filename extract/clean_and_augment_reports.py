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
    # train_1_a_1.nii.gz → train/train_1/train_1_a/train_1_a_1.nii.gz
    base = volume_name.replace(".nii.gz", "")  # train_1_a_1
    parts = base.split("_")  # ['train', '1', 'a', '1']
    folder_lv1 = f"{parts[0]}_{parts[1]}"         # train_1
    folder_lv2 = f"{parts[0]}_{parts[1]}_{parts[2]}"  # train_1_a
    image_path = os.path.join(IMAGE_DIR, folder_lv1, folder_lv2, volume_name)
    return image_path

# 添加图像路径并过滤掉缺失图像的数据
df["image_path"] = df["VolumeName"].apply(get_image_path)
df["image_exists"] = df["image_path"].apply(os.path.exists)
df_filtered = df[df["image_exists"]].drop(columns=["image_exists"])

# 拼接 full_report 字段（Findings_EN + Impressions_EN）
df_filtered["Findings_EN"] = df_filtered["Findings_EN"].fillna("")
df_filtered["Impressions_EN"] = df_filtered["Impressions_EN"].fillna("")
df_filtered["full_report"] = df_filtered["Findings_EN"] + " " + df_filtered["Impressions_EN"]

# 重新整理列顺序或保留字段
df_final = df_filtered[[
    "VolumeName", "image_path", "ClinicalInformation_EN", "Technique_EN",
    "Findings_EN", "Impressions_EN", "full_report"
]]

# 保存新文件
df_final.to_csv("cleaned_train_reports_with_full_text.csv", index=False)

# 查看最终结果数
print(f"原始记录数: {len(df)}")
print(f"清理后（有图像）记录数: {len(df_final)}")