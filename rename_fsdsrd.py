import os
import cv2

# ========== 仅需确认这3个路径是否和你的实际文件夹一致 ==========
# 1. 原始FSDSRD输入图（带阴影）路径
input_im_dir = r"D:\DocRes-master\DocRes-master\data\train\deshadowing\fsdsrd\im"
# 2. 原始FSDSRD真值图（无阴影）路径（如果你的gt在其他文件夹，改这里！）
input_gt_dir = r"D:\DocRes-master\DocRes-master\data\train\deshadowing\fsdsrd\gt"
# 3. 输出路径（和代码读取路径一致，无需改）
output_im_dir = r"D:\DocRes-master\DocRes-master\data\train\deshadowing\fsdsrd\im"
output_gt_dir = r"D:\DocRes-master\DocRes-master\data\train\deshadowing\fsdsrd\gt"

# 创建输出文件夹（防止不存在）
os.makedirs(output_im_dir, exist_ok=True)
os.makedirs(output_gt_dir, exist_ok=True)

# 批量处理输入图：0001.JPG → 0001_in.png
print("正在处理输入图（带阴影）...")
for fname in os.listdir(input_im_dir):
    if fname.endswith(".JPG"):
        # 读取图片
        img_path = os.path.join(input_im_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 跳过无法读取的文件：{img_path}")
            continue
        # 重命名：去掉原文件，避免重复
        new_fname = fname.replace(".JPG", "_in.png")
        new_path = os.path.join(output_im_dir, new_fname)
        # 保存为PNG格式
        cv2.imwrite(new_path, img)
        # 删除原JPG文件（可选，注释掉则保留原文件）
        # os.remove(img_path)

# 批量处理真值图：0001.JPG → 0001_gt.png
print("正在处理真值图（无阴影）...")
for fname in os.listdir(input_gt_dir):
    if fname.endswith(".JPG"):
        img_path = os.path.join(input_gt_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 跳过无法读取的文件：{img_path}")
            continue
        new_fname = fname.replace(".JPG", "_gt.png")
        new_path = os.path.join(output_gt_dir, new_fname)
        cv2.imwrite(new_path, img)
        # 删除原JPG文件（可选）
        # os.remove(img_path)

print("✅ 批量重命名完成！")
print(f"输入图路径：{output_im_dir}（格式：xxx_in.png）")
print(f"真值图路径：{output_gt_dir}（格式：xxx_gt.png）")