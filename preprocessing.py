import cv2
import numpy as np
import pandas as pd
import jieba
import re
import os


class DataPreprocessor:
    def __init__(self, stopwords_path):
        self.stopwords = self.load_stopwords(stopwords_path)

    @staticmethod
    def load_stopwords(stopwords_path):
        stopwords = set()
        if not os.path.exists(stopwords_path):
            print(f"警告：停用词文件 {stopwords_path} 未找到，将使用空集合。")
            return stopwords
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        print(f"成功加载停用词，共{len(stopwords)}个")
        return stopwords

    def read_image(self, img_path, target_size=(256, 256)):
        """
        读取并预处理图像：调整大小、归一化
        """
        if not os.path.exists(img_path):
            print(f"错误：图像文件 {img_path} 不存在")
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"错误：无法读取图像 {img_path}")
            return None

        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 降采样
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        # 归一化到 [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        print(f"图像 {img_path} 处理完成: 原始尺寸 {img_rgb.shape} -> 处后尺寸 {img_normalized.shape}")
        return img_normalized

    def preprocess_text(self, text):
        """
        文本预处理：清洗、分词、去停用词
        """
        if not isinstance(text, str):
            return ""
        # 清洗：保留中文、英文、数字
        clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        # 分词
        words = jieba.lcut(clean_text)
        # 去停用词
        filtered_words = [word for word in words if word not in self.stopwords and word.strip() != ""]
        return " ".join(filtered_words)

    def load_news_data(self, csv_path):
        """
        加载新闻数据，进行文本预处理
        """
        if not os.path.exists(csv_path):
            print(f"错误：新闻数据文件 {csv_path} 不存在")
            return None

        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"新闻数据集总条数：{len(df)}")
        print(f"数据集字段：{df.columns.tolist()}")

        # 预处理文本列（根据实际列名调整）
        # 从错误信息看，列名是 '分类\t分词文章'，需要提取文本部分
        # 假设文本内容在第二个字段（索引为1）
        df['processed_text'] = df['分类\t分词文章'].apply(lambda x: x.split('\t')[1] if '\t' in x else x).apply(self.preprocess_text)
        return df


# 测试代码
if __name__ == "__main__":
    processor = DataPreprocessor("./stopwords.txt")

    # 测试图像读取
    test_img_path = "./BSR/BSDS500/data/images/test/3063.jpg"
    img = processor.read_image(test_img_path)

    # 测试新闻数据加载 - 使用测试数据
    news_csv_path = "./chinese_news_cutted_test_utf8.csv"
    news_df = processor.load_news_data(news_csv_path)

    if img is not None:
        print("图像预处理成功")
    if news_df is not None:
        print("新闻数据预处理成功")
