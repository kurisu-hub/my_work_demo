import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
from preprocessing import DataPreprocessor


class NewsCluster:
    def __init__(self, stopwords_path):
        self.preprocessor = DataPreprocessor(stopwords_path)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.df = None
        self.labels = None
        self.keywords_dict = None
        self.noise_ratio = None

    def load_news_data(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"错误：新闻数据文件 {csv_path} 不存在")
            return None

        # 自动兼容多种编码和分隔符
        encodings = ["utf-8", "utf-8-sig", "gbk"]
        separators = [",", "\t"]
        df = None

        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                    print(f"成功读取CSV，编码：{enc}，分隔符：{repr(sep)}")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            if df is not None:
                break

        if df is None:
            raise ValueError("CSV文件读取失败，请检查文件编码或格式")

        print(f"新闻数据集总条数：{len(df)}")
        print(f"数据集字段（实际列名）：{df.columns.tolist()}")

        # ===================== 通用文本列识别 =====================
        # 1. 获取所有列的数据类型和内容长度统计
        text_col = None
        text_col_stats = {}

        for col in df.columns:
            # 跳过非字符串列
            if not pd.api.types.is_string_dtype(df[col]):
                continue

            # 计算该列的平均字符长度
            avg_length = df[col].dropna().astype(str).str.len().mean()
            text_col_stats[col] = avg_length

        # 2. 选择平均字符长度最长的列作为文本列
        if text_col_stats:
            text_col = max(text_col_stats, key=text_col_stats.get)
            print(f"自动识别文本列：'{text_col}'（平均字符长度：{text_col_stats[text_col]:.1f}）")
        else:
            raise ValueError("未找到合适的文本列，请检查数据格式")

        # 3. 预处理文本列
        df['processed_text'] = df[text_col].apply(self.preprocessor.preprocess_text)
        print("\n文本预处理完成，前3条预览：")
        print(df[[text_col, 'processed_text']].head(3))

        self.df = df
        return df

    def vectorize_news(self, processed_texts=None, max_features=1000):
        # 无参自动调用已加载的预处理数据，兼容两种调用方式
        if processed_texts is None:
            if self.df is None:
                raise ValueError("请先调用load_news_data加载数据")
            processed_texts = self.df['processed_text']

        print(f"开始向量化，最大特征数：{max_features}")
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"向量化完成，矩阵形状：{self.tfidf_matrix.shape}")
        print(f"词汇表大小：{len(self.feature_names)}")
        return self.tfidf_matrix, self.feature_names

    def save_vectorized_data(self, output_dir="./results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, "tfidf_matrix.npy"), self.tfidf_matrix.toarray())

        with open(os.path.join(output_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
            for name in self.feature_names:
                f.write(name + "\n")

        joblib.dump(self.vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
        print(f"向量化数据已保存至 {output_dir}")

    def dbscan_news(self, tfidf_matrix=None, feature_names=None, eps=0.9, min_samples=5):
        """
        使用DBSCAN对新闻文本进行聚类
        """
        if tfidf_matrix is None:
            if self.tfidf_matrix is None:
                raise ValueError("请先调用vectorize_news完成文本向量化")
            tfidf_matrix = self.tfidf_matrix
        if feature_names is None:
            if self.feature_names is None:
                raise ValueError("请先调用vectorize_news完成文本向量化")
            feature_names = self.feature_names

        print(f"开始DBSCAN新闻聚类，eps={eps}, min_samples={min_samples}...")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(tfidf_matrix)

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels) * 100

        print(f"聚类结果：簇数量={n_clusters}, 噪声比例={noise_ratio:.2f}%")

        # 提取每个簇的Top 10关键词（忽略噪声点）
        keywords_dict = self.get_top_keywords(tfidf_matrix, feature_names, labels)

        self.labels = labels
        self.keywords_dict = keywords_dict
        self.noise_ratio = noise_ratio

        return labels, noise_ratio, keywords_dict

    def get_top_keywords(self, tfidf_matrix, feature_names, labels):
        """
        获取每个簇的Top 10关键词（忽略噪声点-1）
        """
        keywords_dict = {}
        unique_labels = set(labels)

        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # 跳过噪声点

            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_tfidf = tfidf_matrix[cluster_indices]
            word_scores = np.mean(cluster_tfidf, axis=0)

            # 处理矩阵格式
            if hasattr(word_scores, 'A1'):
                word_scores = word_scores.A1
            else:
                word_scores = np.asarray(word_scores).flatten()

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

    # 1. 加载向量化数据
    tfidf_matrix = np.load(tfidf_matrix_path)
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f]

    # 2. DBSCAN新闻聚类（调整了参数：eps增大，min_samples减小）
    labels1, noise_ratio1, keywords_dict1 = news_cluster.dbscan_news(tfidf_matrix, feature_names, eps=0.9,
                                                                     min_samples=5)

    # 3. 输出结果
    print(f"\nDBSCAN聚类结果（eps=0.9, min_samples=5）：")
    print(f"噪声比例: {noise_ratio1:.2f}%")
    print("各簇关键词：")
    for cluster_id, keywords in keywords_dict1.items():
        print(f"簇 {cluster_id}: {keywords}")

    print("\n第八天任务完成！")
    print(" - 新闻文本已使用DBSCAN聚类")
    print(" - 噪声比例和关键词已输出")