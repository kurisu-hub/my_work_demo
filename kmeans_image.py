import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from preprocessing import DataPreprocessor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_image_segmentation(test_dir, output_dir="./results", k_values=[3, 5, 8]):
    """
    批量使用K-Means进行图像分割
    :param test_dir: 测试图像目录
    :param output_dir: 输出目录
    :param k_values: K值列表
    """
    processor = DataPreprocessor("./stopwords.txt")

    if not os.path.exists(test_dir):
        print(f"错误：测试目录 {test_dir} 不存在")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取测试目录中的所有图片文件
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"警告：在 {test_dir} 中未找到任何图片文件")
        return

    print(f"找到 {len(image_files)} 张测试图片")

    # 为每张图片创建子目录
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        img_output_dir = os.path.join(output_dir, f"kmeans_{img_name}")
        os.makedirs(img_output_dir, exist_ok=True)

        print(f"\n处理图片: {img_file}")

        # 读取并预处理图像
        img = processor.read_image(img_path)
        if img is None:
            continue

        h, w, c = img.shape
        pixel_values = img.reshape((-1, 3))

        # 为每张图片创建可视化图
        plt.figure(figsize=(15, 5))

        for idx, k in enumerate(k_values):
            print(f"  训练K-Means (K={k})...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixel_values)

            score = silhouette_score(pixel_values, labels)
            print(f"  K={k} 轮廓系数: {score:.4f}")

            centers = np.uint8(kmeans.cluster_centers_ * 255)  # 反归一化
            segmented_img = centers[labels.flatten()].reshape((h, w, c))

            plt.subplot(1, len(k_values), idx + 1)
            plt.imshow(segmented_img)
            plt.title(f"K={k}\nSilhouette: {score:.2f}")
            plt.axis('off')

            # 保存分割结果
            cv2.imwrite(os.path.join(img_output_dir, f"kmeans_k{k}.png"),
                        cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

        plt.tight_layout()
        plt.savefig(os.path.join(img_output_dir, "kmeans_comparison.png"))
        plt.close()

        print(f"  图片处理完成，结果保存在 {img_output_dir}")


if __name__ == "__main__":
    # 测试目录（包含多张图片）
    test_dir = "./BSR/BSDS500/data/images/test/"
    kmeans_image_segmentation(test_dir)
