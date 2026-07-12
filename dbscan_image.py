import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
from preprocessing import DataPreprocessor
from sklearn.cluster import DBSCAN


def dbscan_image_segmentation(test_dir, output_dir="./results"):
    """
    批量使用DBSCAN进行图像分割 (修复版：加入空间特征、平均颜色着色、矩阵化赋值提速)
    """
    processor = DataPreprocessor("./stopwords.txt")
    if not os.path.exists(test_dir):
        print(f"错误：测试目录 {test_dir} 不存在")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"警告：在 {test_dir} 中未找到任何图片文件")
        return

    print(f"找到 {len(image_files)} 张测试图片")

    eps_values = [0.1, 0.15, 0.2]
    min_samples_values = [30, 50, 100]

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        img_output_dir = os.path.join(output_dir, f"dbscan_{img_name}")
        os.makedirs(img_output_dir, exist_ok=True)

        print(f"\n处理图片: {img_file}")
        img = processor.read_image(img_path)
        if img is None:
            continue

        h, w, c = img.shape

        # ========== 核心修复 1：加入空间坐标 ==========
        ys, xs = np.mgrid[0:h, 0:w]
        spatial_weight = 0.1  # 空间权重
        xs_n = (xs / w).astype(np.float32) * spatial_weight
        ys_n = (ys / h).astype(np.float32) * spatial_weight
        features = np.dstack([img, xs_n, ys_n]).reshape((-1, 5))

        results = []
        for eps in eps_values:
            for min_samples in min_samples_values:
                print(f" 测试DBSCAN (eps={eps}, min_samples={min_samples})...")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels = dbscan.fit_predict(features)

                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels) * 100
                print(f" 聚类数: {n_clusters}, 噪声比例: {noise_ratio:.2f}%")

                if 1 < n_clusters < 15:
                    # ========== 核心修复 2：使用簇平均颜色 ==========
                    flat_img = img.reshape(-1, 3)
                    segmented_img = np.zeros((h * w, 3), dtype=np.uint8)

                    for label in unique_labels:
                        if label == -1:
                            segmented_img[labels == label] = [0, 0, 0]
                        else:
                            # 矩阵化赋值，替代原来极慢的双重for循环
                            mean_color = flat_img[labels == label].mean(axis=0)
                            segmented_img[labels == label] = mean_color

                    segmented_img = segmented_img.reshape((h, w, 3))

                    filename = f"dbscan_eps{eps}_min{min_samples}.png"
                    cv2.imwrite(os.path.join(img_output_dir, filename), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
                    results.append({
                        'eps': eps, 'min_samples': min_samples,
                        'n_clusters': n_clusters, 'noise_ratio': noise_ratio,
                        'img': segmented_img
                    })

        if results:
            plt.figure(figsize=(15, 5))
            for i, res in enumerate(results[:3]):
                plt.subplot(1, 3, i + 1)
                plt.imshow(res['img'])
                title = f"Eps:{res['eps']}, Min:{res['min_samples']}\nClusters:{res['n_clusters']}\nNoise:{res['noise_ratio']:.1f}%"
                plt.title(title)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(img_output_dir, "dbscan_results.png"))
            plt.close()
            print(f" 图片处理完成，结果保存在 {img_output_dir}")
        else:
            print(" 未找到有效参数组合，调整eps范围")


if __name__ == "__main__":
    test_dir = "./BSR/BSDS500/data/images/test/"
    dbscan_image_segmentation(test_dir)
