import os
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGE_DIR


def load_image_by_folder(folder_name, file_name):
    """
    加载指定子文件夹下的 NIfTI 文件。
    """
    file_path = os.path.join(IMAGE_DIR, folder_name, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")
    
    img = nib.load(file_path)
    return img.get_fdata()


def show_all_slices(volume, axis=2, step=10):
    """
    可视化 3D CT 图像的多个切片。
    - axis: 显示切片的方向（0/1/2）
    - step: 每隔几张显示一张
    """
    num_slices = volume.shape[axis]
    print(f"🧠 Total slices along axis {axis}: {num_slices}")
    
    for i in range(0, num_slices, step):
        if axis == 0:
            slice_img = volume[i, :, :]
        elif axis == 1:
            slice_img = volume[:, i, :]
        else:
            slice_img = volume[:, :, i]
        
        plt.imshow(slice_img.T, cmap="gray", origin="lower")
        plt.title(f"Axis {axis} - Slice {i}")
        plt.axis("off")
        plt.show()


# ✅ 示例调用
if __name__ == "__main__":
    folder = r"train_1\train_1_a"
    file = "train_1_a_1.nii.gz"
    
    volume = load_image_by_folder(folder, file)
    print("图像维度:", volume.shape)

    # 展示中间切片
    #show_middle_slice(volume)

    # 展示整个 3D 图像（每隔 10 张）
    show_all_slices(volume, axis=2, step=10)
