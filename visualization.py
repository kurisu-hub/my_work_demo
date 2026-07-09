import numpy as np
import matplotlib
# 新增：使用纯离线非交互式后端，不创建Qt窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import jieba
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def plot_text_clusters(tfidf_matrix, labels, feature_names, output_dir="./results", n_components=2, top_n=20):
    """
    可视化文本聚类结果：PCA降维散点图 + 词云图
    :param tfidf_matrix: TF-IDF矩阵（numpy数组）
    :param labels: 簇标签（numpy数组）
    :param feature_names: 特征名称列表（词汇表）
    :param output_dir: 输出目录
    :param n_components: PCA降维维度（默认2）
    :param top_n: 词云关键词数量（默认20）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. PCA降维
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(tfidf_matrix)  # 直接使用密集矩阵
    print(f"PCA降维完成，保留方差比例：{np.sum(pca.explained_variance_ratio_):.2%}")

    # 2. 绘制散点图（修改：使用 plt.get_cmap 代替 plt.cm.get_cmap）
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    colors = plt.get_cmap('tab10', len(unique_labels))  # 修改这里

    for cluster_id in unique_labels:
        # 获取该簇的索引
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # 获取该簇的降维数据
        cluster_data = reduced_data[cluster_indices]

        # 绘制散点
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=colors(cluster_id), label=f"簇 {cluster_id}", alpha=0.7)

    plt.title("文本聚类结果（PCA降维）")
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "text_clusters.png"))
    plt.close()
    print(f"散点图已保存至 {output_dir}/text_clusters.png")

    # 3. 生成词云图
    for cluster_id in unique_labels:
        # 获取该簇的索引
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # 获取该簇的TF-IDF矩阵
        cluster_tfidf = tfidf_matrix[cluster_indices]

        # 计算每个词的平均TF-IDF值
        word_scores = np.mean(cluster_tfidf, axis=0)

        # 获取Top N关键词
        top_indices = np.argsort(word_scores)[::-1][:top_n]
        top_keywords = [feature_names[i] for i in top_indices]
        top_scores = [word_scores[i] for i in top_indices]

        # 生成词云
        wordcloud = WordCloud(
            font_path="simhei.ttf",  # 中文支持，需下载simhei.ttf字体文件
            width=800,
            height=400,
            background_color="white"
        ).generate_from_frequencies(dict(zip(top_keywords, top_scores)))

        # 保存词云图
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"簇 {cluster_id} 词云图")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_wordcloud.png"))
        plt.close()
        print(f"簇 {cluster_id} 词云图已保存至 {output_dir}/cluster_{cluster_id}_wordcloud.png")


if __name__ == "__main__":
    # 路径配置
    tfidf_matrix_path = "./results/tfidf_matrix.npy"
    labels_path = "./results/labels_kmeans.npy"
    feature_names_path = "./results/feature_names.txt"

    # 加载向量化数据和簇标签
    tfidf_matrix = np.load(tfidf_matrix_path)
    labels = np.load(labels_path)
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f]

    # 可视化文本聚类结果
    plot_text_clusters(tfidf_matrix, labels, feature_names)

    print(" - 文本聚类结果已可视化（散点图+词云图）")
    print(" - 结果已保存至 ./results/ 目录")
