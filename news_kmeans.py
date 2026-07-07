import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import joblib


from preprocessing import DataPreprocessor


class NewsCluster:
    def __init__(self, stopwords_path):
        self.preprocessor = DataPreprocessor(stopwords_path)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

    def load_news_data(self, csv_path):
        """加载新闻数据并进行文本预处理"""
        if not os.path.exists(csv_path):
            print(f"错误：新闻数据文件 {csv_path} 不存在")
            return None

        df = pd.read_csv(csv_path, encoding="gbk", sep="\t")
        print(f"新闻数据集总条数：{len(df)}")
        print(f"数据集字段（实际列名）：{df.columns.tolist()}")


        if 'test' not in df.columns:
            print("错误：CSV 文件中未找到 'test' 列")
            return None

        # 使用预处理类处理文本
        df['processed_text'] = df['test'].apply(self.preprocessor.preprocess_text)
        print("\n文本预处理完成，前3条预览：")
        print(df[['test', 'processed_text']].head(3))

        return df

    def vectorize_news(self, processed_texts, max_features=500):
        """使用 TF-IDF 向量化新闻文本"""
        print(f"开始向量化，最大特征数：{max_features}")

        # 初始化 TF-IDF 向量化器
        self.vectorizer = TfidfVectorizer(max_features=max_features)

        # 拟合并转换文本
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"向量化完成，矩阵形状：{self.tfidf_matrix.shape}")
        print(f"词汇表大小：{len(self.feature_names)}")

        return self.tfidf_matrix, self.feature_names

    def save_vectorized_data(self, output_dir="./results"):
        """保存向量化后的数据和特征名称"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存 TF-IDF 矩阵（.npy 格式）
        np.save(os.path.join(output_dir, "tfidf_matrix.npy"), self.tfidf_matrix.toarray())

        # 保存特征名称（.txt 格式）
        with open(os.path.join(output_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
            for name in self.feature_names:
                f.write(name + "\n")

        # 保存向量化器（用于后续预测）
        joblib.dump(self.vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))

        print(f"向量化数据已保存至 {output_dir}")

    def kmeans_news(self, tfidf_matrix, feature_names, k=5):
        """
        使用K-Means对新闻文本进行聚类
        :param tfidf_matrix: TF-IDF矩阵（numpy数组）
        :param feature_names: 特征名称列表（词汇表）
        :param k: 聚类数（默认5）
        :return: 簇标签，轮廓系数，关键词字典
        """
        print(f"开始K-Means新闻聚类，K={k}...")

        # 模型训练
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)

        # 计算轮廓系数
        score = silhouette_score(tfidf_matrix, labels)
        print(f"K={k} 时的轮廓系数: {score:.4f}")

        # 提取每个簇的Top 10关键词
        keywords_dict = self.get_top_keywords(tfidf_matrix, feature_names, labels, k)
        np.save("./results/labels_kmeans.npy", labels)
        print("簇标签已保存至 ./results/labels_kmeans.npy")
        return labels, score, keywords_dict

    def get_top_keywords(self, tfidf_matrix, feature_names, labels, k):
        """
        获取每个簇的Top 10关键词
        :param tfidf_matrix: TF-IDF矩阵
        :param feature_names: 特征名称
        :param labels: 簇标签
        :param k: 聚类数
        :return: 关键词字典（键：簇编号，值：关键词列表）
        """
        keywords_dict = {}
        for cluster_id in range(k):
            # 获取该簇的文本索引
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                keywords_dict[cluster_id] = []
                continue

            # 获取该簇的TF-IDF矩阵
            cluster_tfidf = tfidf_matrix[cluster_indices]

            # 计算每个词在该簇的平均TF-IDF值
            word_scores = np.mean(cluster_tfidf, axis=0)

            # 获取Top 10关键词的索引
            top_indices = np.argsort(word_scores)[::-1][:10]
            top_keywords = [feature_names[i] for i in top_indices]

            keywords_dict[cluster_id] = top_keywords

        return keywords_dict



if __name__ == "__main__":
    # 路径配置
    stopwords_path = "./stopwords.txt"
    news_csv_path = "./chinese_news_cutted_test_utf8.csv"
    tfidf_matrix_path = "./results/tfidf_matrix.npy"
    feature_names_path = "./results/feature_names.txt"

    # 初始化新闻聚类类
    news_cluster = NewsCluster(stopwords_path)

    # 1. 加载新闻数据并预处理（若未预处理，可调用load_news_data和vectorize_news）
    # 这里假设已预处理并保存向量化数据，直接加载
    tfidf_matrix = np.load(tfidf_matrix_path)
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f]

    # 2. K-Means新闻聚类
    labels, score, keywords_dict = news_cluster.kmeans_news(tfidf_matrix, feature_names, k=5)

    # 3. 输出结果
    print(f"\nK-Means聚类结果：")
    print(f"轮廓系数: {score:.4f}")
    print("各簇关键词：")
    for cluster_id, keywords in keywords_dict.items():
        print(f"簇 {cluster_id}: {keywords}")

    print("\n第七天任务完成！")
    print(" - 新闻文本已使用K-Means聚类")
    print(" - 轮廓系数和关键词已输出")
