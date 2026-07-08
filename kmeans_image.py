import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from preprocessing import DataPreprocessor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_image_segmentation(input_path, output_dir="./results", k_values=[3, 5, 8]):
    """
    K-Means图像分割，支持单张图片或图片目录
    :param input_path: 单张图片路径 或 图片目录路径
    :param output_dir: 输出目录
    :param k_values: K值列表
    :return: 包含轮廓系数和输出文件路径的字典
    """
    processor = DataPreprocessor("./stopwords.txt")

    if not os.path.exists(input_path):
        print(f"错误：路径 {input_path} 不存在")
        return {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 判断输入是文件还是目录
    if os.path.isfile(input_path):
        image_files = [os.path.basename(input_path)]
        input_dir = os.path.dirname(input_path)
    else:
        input_dir = input_path
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"警告：在 {input_path} 中未找到任何图片文件")
        return {}

    print(f"找到 {len(image_files)} 张图片")
    all_results = {}

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
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

        plt.figure(figsize=(15, 5))
        output_files = []
        silhouette_scores = {}

        for idx, k in enumerate(k_values):
            print(f"  训练K-Means (K={k})...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixel_values)

            score = silhouette_score(pixel_values, labels)
            silhouette_scores[k] = score
            print(f"  K={k} 轮廓系数: {score:.4f}")

            centers = np.uint8(kmeans.cluster_centers_ * 255)
            segmented_img = centers[labels.flatten()].reshape((h, w, c))

            plt.subplot(1, len(k_values), idx + 1)
            plt.imshow(segmented_img)
            plt.title(f"K={k}\nSilhouette: {score:.2f}")
            plt.axis('off')

            save_path = os.path.join(img_output_dir, f"kmeans_k{k}.png")
            cv2.imwrite(save_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
            output_files.append(save_path)

        plt.tight_layout()
        plt.savefig(os.path.join(img_output_dir, "kmeans_comparison.png"))
        plt.close()

        print(f"  处理完成，结果保存在 {img_output_dir}")
        all_results[img_name] = {
            'silhouette_score': silhouette_scores,
            'output_files': output_files,
            'output_dir': img_output_dir
        }

    # 单图模式下直接返回第一个结果，适配GUI调用
    if os.path.isfile(input_path):
        first_key = list(all_results.keys())[0]
        res = all_results[first_key]
        # 取第一个K值的轮廓系数，兼容GUI显示
        res['silhouette_score'] = res['silhouette_score'][k_values[0]]
        return res

    return all_results
if __name__ == "__main__":
    # 测试目录（包含多张图片）
    test_dir = "./BSR/BSDS500/data/images/test/"
    kmeans_image_segmentation(test_dir)
