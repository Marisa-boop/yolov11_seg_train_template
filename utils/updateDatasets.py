import os
import shutil


def incremental_update(merged_data_dir="merged_data", added_data_dir="added_data"):
    """
    增量更新图像和标签文件到目标目录，自动处理重名冲突
    保持图像和标签文件的命名一致性（同名文件不同扩展名）

    :param merged_data_dir: 目标数据目录（包含images/labels子目录）
    :param added_data_dir: 新增数据目录（包含images/labels子目录）
    """
    # 定义目录路径
    src_images_dir = os.path.join(added_data_dir, "images")
    src_labels_dir = os.path.join(added_data_dir, "labels")
    dest_images_dir = os.path.join(merged_data_dir, "images")
    dest_labels_dir = os.path.join(merged_data_dir, "labels")

    # 确保目标目录存在
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    # 获取新增数据文件列表
    src_images = [
        f
        for f in os.listdir(src_images_dir)
        if os.path.isfile(os.path.join(src_images_dir, f))
    ]

    # 处理每个图像文件及其对应的标签文件
    for img_file in src_images:
        # 提取文件名和扩展名
        img_name, img_ext = os.path.splitext(img_file)
        label_file = f"{img_name}.txt"  # 假设标签文件是.txt格式

        # 构建完整源路径
        src_img_path = os.path.join(src_images_dir, img_file)
        src_label_path = os.path.join(src_labels_dir, label_file)

        # 验证标签文件是否存在（确保一一对应）
        if not os.path.exists(src_label_path):
            print(f"⚠️ 标签文件缺失: {label_file}，跳过处理")
            continue

        # 构建目标路径（初始）
        dest_img_path = os.path.join(dest_images_dir, img_file)
        dest_label_path = os.path.join(dest_labels_dir, label_file)

        # 检查目标文件是否存在
        if not (os.path.exists(dest_img_path) or os.path.exists(dest_label_path)):
            # 无冲突，直接复制
            shutil.copy2(src_img_path, dest_img_path)
            shutil.copy2(src_label_path, dest_label_path)
            print(f"✅ 已添加: {img_file} 和 {label_file}")
        else:
            # 处理冲突（生成唯一文件名）
            counter = 1
            while True:
                new_img_name = f"{img_name}_{counter}{img_ext}"
                new_label_name = f"{img_name}_{counter}.txt"

                new_img_path = os.path.join(dest_images_dir, new_img_name)
                new_label_path = os.path.join(dest_labels_dir, new_label_name)

                # 检查新文件名是否可用
                if not os.path.exists(new_img_path) and not os.path.exists(new_label_path):
                    shutil.copy2(src_img_path, new_img_path)
                    shutil.copy2(src_label_path, new_label_path)
                    print(f"🔄 已添加（重命名）: {img_file} -> {new_img_name}, {label_file} -> {new_label_name}")
                    break
                counter += 1


if __name__ == "__main__":
    # 执行增量更新
    incremental_update()

    print("=" * 50)
    print("🛠️ 增量更新完成")
    print(f"目标目录: {os.path.abspath('merged_data')}")
    print(f"新增数据: {os.path.abspath('added_data')}")
    print("=" * 50)
