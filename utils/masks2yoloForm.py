from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
import cv2
import numpy as np
from pathlib import Path
import os


def normalized_masks(input_dir, output_dir):
    # 遍历所有掩码文件
    for mask_file in Path(input_dir).glob("*.png"):
        # 读取掩码图像
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        # 获取非零唯一像素值（排除背景0）
        unique_vals = np.unique(mask)
        unique_vals = unique_vals[unique_vals != 0]

        # 创建重映射字典
        remap_dict = {old_val: new_val for new_val,
                      old_val in enumerate(unique_vals, start=1)}

        # 应用像素值重映射
        for old_val, new_val in remap_dict.items():
            mask[mask == old_val] = new_val

        # 保存处理后的掩码
        output_path = Path(output_dir) / mask_file.name
        cv2.imwrite(str(output_path), mask)
        print(f"处理完成: {mask_file} → {output_path}")

    print("所有掩码文件处理完毕！")


if __name__ == "__main__":
    # 创建输出目录
    base_dir = "./merged_data"
    input_dir = "./merged_data/masks"
    output_dir = "./merged_data/masks_normalized"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels_dir = "./merged_data/labels"
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    normalized_masks(input_dir, output_dir)
    # The classes here is the total classes in the dataset.
    # for COCO dataset we have 80 classes.
    convert_segment_masks_to_yolo_seg(
        masks_dir="./merged_data/masks_normalized", output_dir="./merged_data/labels", classes=6)
