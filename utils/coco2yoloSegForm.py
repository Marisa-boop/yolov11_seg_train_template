import json
import os
import argparse
from tqdm import tqdm
import numpy as np

def convert_coco_to_yolo_segment(json_path: str, output_dir: str):
    """
    将 COCO 格式的分割标注转换为 YOLO 实例分割格式
    :param json_path: COCO 格式的 JSON 文件路径
    :param output_dir: YOLO 标签输出目录
    """
    # 加载 JSON 数据
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建图像 ID 到图像元数据的映射
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # 类别映射表 (COCO ID → YOLO 0-based ID)
    categories = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    
    # 遍历所有标注
    for annotation in tqdm(coco_data['annotations'], desc="Converting annotations"):
        img_id = annotation['image_id']
        if img_id not in images_dict:
            continue  # 跳过无效图像ID
        
        # 获取图像信息
        img_info = images_dict[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        img_name = os.path.splitext(img_info['file_name'])[0]
        
        # 准备输出文件路径
        txt_path = os.path.join(output_dir, f"{img_name}.txt")
        
        # 处理分割标注
        segmentation = annotation.get('segmentation', [])
        if not segmentation:
            continue  # 跳过无分割标注的对象
        
        # 获取类别ID (YOLO格式为0-based)
        class_id = categories.get(annotation['category_id'], -1)
        if class_id == -1:
            continue  # 跳过无效类别
        
        # 转换坐标格式
        yolo_segments = []
        for segment in segmentation:
            # 将坐标点分组为 [x1, y1, x2, y2, ...]
            points = np.array(segment).reshape(-1, 2)
            # 坐标归一化 (0-1范围)
            normalized_points = [
                f"{x / img_width:.6f} {y / img_height:.6f}"
                for x, y in points
            ]
            yolo_segments.extend(normalized_points)
        
        # 写入YOLO格式
        with open(txt_path, 'a') as f:
            line = f"{class_id} " + " ".join(yolo_segments) + "\n"
            f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='COCO to YOLO Segment Format Converter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--json', type=str, required=True,
                        help='COCO格式JSON文件路径')
    parser.add_argument('--output', type=str, default='labels',
                        help='YOLO标签输出目录')
    args = parser.parse_args()

    convert_coco_to_yolo_segment(args.json, args.output)
    print(f"✅ 转换完成！标签已保存至: {args.output}")
