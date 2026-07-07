import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from preprocessing import DataPreprocessor
from sklearn.cluster import DBSCAN


def dbscan_image_segmentation(test_dir, output_dir="./results"):
    """
    批量使用DBSCAN进行图像分割
    :param test_dir: 测试图像目录
    :param output_dir: 输出目录
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

    # 参数组合
    eps_values = [0.05, 0.1, 0.15, 0.2]
    min_samples_values = [50, 100, 200]

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        img_output_dir = os.path.join(output_dir, f"dbscan_{img_name}")
        os.makedirs(img_output_dir, exist_ok=True)

        print(f"\n处理图片: {img_file}")

        # 读取并预处理图像
        img = processor.read_image(img_path)
        if img is None:
            continue

        h, w, c = img.shape
        pixel_values = img.reshape((-1, 3))

        results = []

        for eps in eps_values:
            for min_samples in min_samples_values:
                print(f"  测试DBSCAN (eps={eps}, min_samples={min_samples})...")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels = dbscan.fit_predict(pixel_values)

                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels) * 100

                print(f"  聚类数: {n_clusters}, 噪声比例: {noise_ratio:.2f}%")

                if 1 < n_clusters < 15:
                    segmented_img = np.zeros((h, w, c), dtype=np.uint8)
                    for i in range(h):
                        for j in range(w):
                            label = labels[i * w + j]
                            if label == -1:
                                segmented_img[i, j] = [0, 0, 0]  # 噪声为黑色
                            else:
                                color = np.random.randint(0, 256, 3)
                                segmented_img[i, j] = color

                    filename = f"dbscan_eps{eps}_min{min_samples}.png"
                    cv2.imwrite(os.path.join(img_output_dir, filename),
                                cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

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
            plt.savefig(os.path.join(img_output_dir, "dbscan_results.png"))
            plt.close()

            print(f"  图片处理完成，结果保存在 {img_output_dir}")
        else:
            print("  未找到有效参数组合，调整eps范围")


if __name__ == "__main__":
    # 测试目录（包含多张图片）
    test_dir = "./BSR/BSDS500/data/images/test/"
    dbscan_image_segmentation(test_dir)
