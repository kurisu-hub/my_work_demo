import pandas as pd
import numpy as np
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

    def load_news_data(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"错误：新闻数据文件 {csv_path} 不存在")
            return None

        # 修改编码为 gbk（或 gb2312）
        df = pd.read_csv(csv_path, encoding="gbk", sep="\t")
        print(f"新闻数据集总条数：{len(df)}")
        print(f"数据集字段（实际列名）：{df.columns.tolist()}")

        if 'test' not in df.columns:
            print("错误：CSV 文件中未找到 'test' 列")
            return None

        df['processed_text'] = df['test'].apply(self.preprocessor.preprocess_text)
        print("\n文本预处理完成，前3条预览：")
        print(df[['test', 'processed_text']].head(3))

        return df

    def vectorize_news(self, processed_texts, max_features=1000):
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


if __name__ == "__main__":
    stopwords_path = "./stopwords.txt"
    news_csv_path = "./chinese_news_cutted_test_utf8.csv"
    news_cluster = NewsCluster(stopwords_path)
    news_df = news_cluster.load_news_data(news_csv_path)
    if news_df is None:
        print("新闻数据加载失败，请检查路径或数据格式")
        exit()
    processed_texts = news_df['processed_text'].tolist()
    tfidf_matrix, feature_names = news_cluster.vectorize_news(processed_texts, max_features=500)
    news_cluster.save_vectorized_data()
    print(" - 新闻数据已加载并预处理")
    print(" - 文本已向量化为 TF-IDF 矩阵")
    print(" - 数据和特征已保存至 ./results/ 目录")
