import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
import os
from preprocessing import DataPreprocessor


def dbscan_image_segmentation(image_path, output_dir="./results"):
    """
    使用DBSCAN进行图像分割
    """
    processor = DataPreprocessor("./stopwords.txt")
    img = processor.read_image(image_path)

    if img is None:
        return

    h, w, c = img.shape
    pixel_values = img.reshape((-1, 3))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eps_values = [0.05, 0.1, 0.15, 0.2]
    min_samples_values = [50, 100, 200]

    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"测试DBSCAN (eps={eps}, min_samples={min_samples})...")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(pixel_values)

            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels) * 100

            print(f"聚类数: {n_clusters}, 噪声比例: {noise_ratio:.2f}%")

            if 1 < n_clusters < 15:
                segmented_img = np.zeros((h, w, c), dtype=np.uint8)  # 初始化为uint8类型
                for i in range(h):
                    for j in range(w):
                        label = labels[i * w + j]
                        if label == -1:
                            segmented_img[i, j] = [0, 0, 0]  # 噪声为黑色
                        else:
                            # 为每个簇分配随机颜色（0-255整数）
                            color = np.random.randint(0, 256, 3)
                            segmented_img[i, j] = color

                filename = f"dbscan_eps{eps}_min{min_samples}.png"
                cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

                results.append({
                    'eps': eps, 'min_samples': min_samples,
                    'n_clusters': n_clusters, 'noise_ratio': noise_ratio,
                    'img': segmented_img
                })

    if results:
        plt.figure(figsize=(15, 5))
        for i, res in enumerate(results[:3]):  # 展示前3个结果
            plt.subplot(1, 3, i + 1)
            plt.imshow(res['img'])
            title = f"Eps:{res['eps']}, Min:{res['min_samples']}\nClusters:{res['n_clusters']}\nNoise:{res['noise_ratio']:.1f}%"
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dbscan_results.png"))
        plt.show()
    else:
        print("未找到有效参数组合，调整eps范围")


# 测试代码
if __name__ == "__main__":
    test_img = "./BSR/BSDS500/data/images/test/3063.jpg"
    if os.path.exists(test_img):
        dbscan_image_segmentation(test_img)
    else:
        print(f"请确认图片路径 {test_img} 存在")
