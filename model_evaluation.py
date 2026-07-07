import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
import os
import sys
from datetime import datetime

# 导入自定义的预处理模块
try:
    from preprocessing import DataPreprocessor
except ImportError:
    print("错误：未找到 preprocessing.py，请确保该模块在项目根目录下。")
    sys.exit(1)


def evaluate_clustering(X, labels, algorithm_name="Algorithm"):
    """
    评估聚类结果的通用函数
    :param X: 特征矩阵
    :param labels: 聚类标签
    :param algorithm_name: 算法名称
    :return: 包含评估指标的字典
    """
    results = {
        "Algorithm": algorithm_name,
        "Noise Ratio": 0,
        "Silhouette Score": "N/A",
        "Davies-Bouldin Index": "N/A",
        "Calinski-Harabasz Index": "N/A",
        "Number of Clusters": "N/A"
    }

    # 统计噪声点和有效簇数
    noise_count = np.sum(labels == -1)
    results["Noise Ratio"] = f"{noise_count / len(labels) * 100:.2f}%"

    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    unique_labels = np.unique(labels_valid)
    results["Number of Clusters"] = len(unique_labels)

    # 只有当簇数大于1且小于样本数时才能计算指标
    if len(unique_labels) > 1 and len(unique_labels) < X_valid.shape[0]:
        try:
            sil_score = metrics.silhouette_score(X_valid, labels_valid)
            results["Silhouette Score"] = f"{sil_score:.4f}"
        except:
            results["Silhouette Score"] = "Error"

        try:
            db_score = metrics.davies_bouldin_score(X_valid, labels_valid)
            results["Davies-Bouldin Index"] = f"{db_score:.4f}"
        except:
            results["Davies-Bouldin Index"] = "Error"

        try:
            ch_score = metrics.calinski_harabasz_score(X_valid, labels_valid)
            results["Calinski-Harabasz Index"] = f"{ch_score:.4f}"
        except:
            results["Calinski-Harabasz Index"] = "Error"
    else:
        results["Note"] = f"Insufficient clusters ({len(unique_labels)}) for metrics"

    return results


def generate_detailed_report(results_list):
    """
    生成详细的评估报告
    :param results_list: 评估结果列表
    :return: 格式化的报告字符串
    """
    if not results_list:
        return "未生成任何评估结果。"

    df_results = pd.DataFrame(results_list)

    # 生成报告头部
    report = []
    report.append("=" * 80)
    report.append("聚类算法综合评估报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    # 添加指标说明
    report.append("\n指标说明:")
    report.append("- Silhouette Score: 范围[-1, 1]，越接近1效果越好。")
    report.append("- Davies-Bouldin Index: 最小值为0，越小越好。")
    report.append("- Calinski-Harabasz Index: 无上界，越大越好。")
    report.append("- Noise Ratio: DBSCAN特有的噪声点比例。")
    report.append("=" * 80)

    # 添加结果表格
    report.append("\n评估结果:")
    report.append(df_results.to_string(index=False))
    report.append("=" * 80)

    # 添加结果分析
    report.append("\n结果分析:")
    for result in results_list:
        analysis = []
        analysis.append(f"\n{result['Algorithm']}:")

        # 轮廓系数分析
        sil_score = result['Silhouette Score']
        if sil_score != "N/A" and sil_score != "Error":
            score = float(sil_score)
            if score > 0.5:
                analysis.append(f"  - 轮廓系数({score:.4f})：聚类效果优秀")
            elif score > 0.3:
                analysis.append(f"  - 轮廓系数({score:.4f})：聚类效果良好")
            elif score > 0.1:
                analysis.append(f"  - 轮廓系数({score:.4f})：聚类效果一般")
            else:
                analysis.append(f"  - 轮廓系数({score:.4f})：聚类效果较差")

        # DB指数分析
        db_score = result['Davies-Bouldin Index']
        if db_score != "N/A" and db_score != "Error":
            score = float(db_score)
            if score < 1:
                analysis.append(f"  - DB指数({score:.4f})：簇间分离度好")
            elif score < 2:
                analysis.append(f"  - DB指数({score:.4f})：簇间分离度一般")
            else:
                analysis.append(f"  - DB指数({score:.4f})：簇间分离度差")

        # 噪声比例分析（仅DBSCAN）
        if "DBSCAN" in result['Algorithm']:
            noise_ratio = result['Noise Ratio']
            if noise_ratio != "0":
                analysis.append(f"  - 噪声比例({noise_ratio})：数据中存在较多噪声点")

        report.extend(analysis)

    report.append("=" * 80)
    return "\n".join(report)


def main():
    results_list = []
    output_dir = "./results"
    stopwords_path = "./stopwords.txt"

    # 初始化预处理器
    preprocessor = DataPreprocessor(stopwords_path)

    # ==========================================
    # 1. 图像聚类评估
    # ==========================================
    print("正在评估图像聚类...")
    test_img_path = "./BSR/BSDS500/data/images/test/3063.jpg"

    if os.path.exists(test_img_path):
        # 1.1 加载并预处理图像
        img = preprocessor.read_image(test_img_path)
        pixels = img.reshape(-1, 3)  # 展平为 (N, 3)

        # 1.2 评估 K-Means (K=5)
        kmeans_img = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels_kmeans_img = kmeans_img.fit_predict(pixels)
        results_list.append(evaluate_clustering(pixels, labels_kmeans_img, "Image K-Means (K=5)"))

        # 1.3 评估 DBSCAN (eps=0.1, min_samples=100)
        dbscan_img = DBSCAN(eps=0.1, min_samples=100, n_jobs=-1)
        labels_dbscan_img = dbscan_img.fit_predict(pixels)
        results_list.append(evaluate_clustering(pixels, labels_dbscan_img, "Image DBSCAN (eps=0.1)"))
    else:
        print("警告：未找到图像测试文件，跳过图像评估。")

    # ==========================================
    # 2. 文本聚类评估
    # ==========================================
    print("正在评估文本聚类...")
    tfidf_path = os.path.join(output_dir, "tfidf_matrix.npy")
    labels_km_path = os.path.join(output_dir, "labels_kmeans.npy")
    labels_db_path = os.path.join(output_dir, "labels_dbscan.npy")

    if os.path.exists(tfidf_path):
        tfidf_matrix = np.load(tfidf_path)

        # 2.1 评估 K-Means (加载保存的标签)
        if os.path.exists(labels_km_path):
            labels_kmeans_txt = np.load(labels_km_path)
            results_list.append(evaluate_clustering(tfidf_matrix, labels_kmeans_txt, "Text K-Means (K=5)"))
        else:
            print("警告：未找到 labels_kmeans.npy，跳过文本K-Means评估。")

        # 2.2 评估 DBSCAN
        if os.path.exists(labels_db_path):
            # 如果已有保存的标签，直接加载
            labels_dbscan_txt = np.load(labels_db_path)
            results_list.append(evaluate_clustering(tfidf_matrix, labels_dbscan_txt, "Text DBSCAN (eps=0.9)"))
        else:
            # 如果没有保存，则临时跑一次（根据Day 8的最佳参数）
            print("未找到 labels_dbscan.npy，正在临时计算...")
            dbscan_txt = DBSCAN(eps=0.9, min_samples=5, n_jobs=-1)
            labels_dbscan_txt = dbscan_txt.fit_predict(tfidf_matrix)
            # 顺便保存一下，方便下次使用
            np.save(os.path.join(output_dir, "labels_dbscan.npy"), labels_dbscan_txt)
            results_list.append(evaluate_clustering(tfidf_matrix, labels_dbscan_txt, "Text DBSCAN (eps=0.9)"))
    else:
        print("警告：未找到 tfidf_matrix.npy，跳过文本评估。")

    # ==========================================
    # 3. 生成评估报告
    # ==========================================
    if results_list:
        # 生成详细报告
        report_content = generate_detailed_report(results_list)

        # 打印到控制台
        print(report_content)

        # 保存到文件
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\n评估报告已保存至：{report_path}")
    else:
        print("未生成任何评估结果。")


if __name__ == "__main__":
    main()
