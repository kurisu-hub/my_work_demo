import cv2
import numpy as np
import pandas as pd
import jieba

#
def load_stopwords(stopwords_path):
    stopwords = set()
    with open(stopwords_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    print(f"成功加载停用词，共{len(stopwords)}个")
    return stopwords

#
def read_news_csv(csv_path, stopwords):
    # 读取csv，第一行作为列名，按制表符拆分列
    df = pd.read_csv(csv_path, encoding="utf-8", sep="\t", header=0)
    print(f"新闻数据集总条数：{len(df)}")
    print(f"数据集字段：{df.columns.tolist()}")
    # 预览前3条有效数据
    print("\n前3条新闻数据预览：")
    print(df.head(3))
    # 停用词过滤测试
    sample_text = df["分词文章"].iloc[0]
    cut_words = jieba.lcut(sample_text)
    filtered_words = [word for word in cut_words if word not in stopwords and word.strip() != ""]
    print(f"\n单条新闻原始分词结果：{cut_words[:15]}")
    print(f"单条新闻停用词过滤后结果：{filtered_words[:15]}")
    return df

# 图像读取+降采样预处理
def read_and_process_image(img_path, target_size=(300, 300)):
    img = cv2.imread(img_path)
    if img is None:
        print("错误：图像文件不存在，请检查路径或图片文件")
        return None
    # 色彩通道转换
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 降采样预处理，压缩尺寸提升聚类速度
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    print(f"原始图像尺寸：{img_rgb.shape}")
    print(f"降采样后图像尺寸：{img_resized.shape}")
    return img_resized


if __name__ == "__main__":
    # 路径配置
    stopwords_path = "./stopwords.txt"
    train_csv_path = "./chinese_news_cutted_train_utf8.csv"
    test_csv_path = "./chinese_news_cutted_test_utf8.csv"
    test_img_path = "./img/test.jpg"

    # 加载停用词
    stopwords = load_stopwords(stopwords_path)
    # 读取训练集、测试集
    print("\n===== 训练集读取测试 =====")
    df_train = read_news_csv(train_csv_path, stopwords)
    print("\n===== 测试集读取测试 =====")
    df_test = read_news_csv(test_csv_path, stopwords)
    # 图像读取测试
    print("\n===== 图像读取测试 =====")
    img = read_and_process_image(test_img_path)