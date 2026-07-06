import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os
from preprocessing import DataPreprocessor


def kmeans_image_segmentation(image_path, output_dir="./results", k_values=[3, 5, 8]):
    """
    使用K-Means进行图像分割
    """
    processor = DataPreprocessor("./stopwords.txt")
    img = processor.read_image(image_path)

    if img is None:
        return

    h, w, c = img.shape
    pixel_values = img.reshape((-1, 3))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(15, 5))

    for idx, k in enumerate(k_values):
        print(f"训练K-Means (K={k})...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixel_values)

        score = silhouette_score(pixel_values, labels)
        print(f"K={k} 轮廓系数: {score:.4f}")

        centers = np.uint8(kmeans.cluster_centers_ * 255)  # 反归一化
        segmented_img = centers[labels.flatten()].reshape((h, w, c))

        plt.subplot(1, len(k_values), idx + 1)
        plt.imshow(segmented_img)
        plt.title(f"K={k}\nSilhouette: {score:.2f}")
        plt.axis('off')

        cv2.imwrite(os.path.join(output_dir, f"kmeans_k{k}.png"), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_comparison.png"))
    plt.show()


# 测试代码
if __name__ == "__main__":
    test_img = "./BSR/BSDS500/data/images/test/3063.jpg"
    if os.path.exists(test_img):
        kmeans_image_segmentation(test_img)
    else:
        print(f"请确认图片路径 {test_img} 存在")
