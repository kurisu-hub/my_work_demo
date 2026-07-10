# 图像分割与新闻文本聚类综合应用系统

## 项目简介
本项目为人工智能综合实践课程设计项目，基于 Python + Scikit-learn + PyQt5 桌面交互界面完成无监督聚类系统，实现两大核心功能：
- 基于 K-Means、DBSCAN 算法完成图像像素级分割
- 中文新闻文本预处理、TF-IDF 向量化，两种算法完成新闻主题聚类
- 使用轮廓系数完成聚类效果量化评估，支持 PCA 降维可视化
- 纯代码 PyQt5 交互式桌面界面，无需 Qt Designer 拖拽，一键运行

对应课程设计要求全部覆盖：
- 数据预处理：图像像素归一化、中文分词、停用词过滤、文本向量化
- 算法实现：sklearn 封装 Kmeans/DBSCAN 聚类
- 性能评估：轮廓系数作为聚类标准指标
- 可视化：图像分割结果、PCA 聚类散点图
- 交互式界面：PyQt5 桌面 GUI，参数调节、文件上传、结果实时展示
- 完整数据读取、模型训练、结果输出全流程

---

## 开发环境
- 操作系统：Windows 11
- IDE：PyCharm 2023+
- Python 版本：3.12.2
- GUI 框架：PyQt5

---

## 依赖安装
推荐新建虚拟环境后执行安装命令：
```bash
pip install -r requirements.txt
项目目录说明
plaintext
my_work_demo/
├── main.py                     # 程序入口，PyQt5交互界面主程序
├── kmeans_image.py             # K-Means 图像分割算法实现
├── dbscan_image.py             # DBSCAN 图像分割算法实现
├── news_cluster.py             # 新闻聚类业务逻辑、文件上传与校验
├── news_kmeans.py              # 新闻文本聚类算法（K-Means/DBSCAN）
├── preprocessing.py            # 数据预处理模块（图像/文本）
├── model_evaluation.py         # 聚类评估指标（轮廓系数）计算
├── visualization.py            # PCA降维、聚类结果可视化
├── test.py                     # 功能测试脚本
├── requirements.txt            # 项目依赖包清单
├── stopwords.txt               # 中文停用词词典
├── chinese_news_cutted_test_utf8.csv    # 中文新闻测试数据集
├── chinese_news_cutted_train_utf8.csv   # 中文新闻训练数据集
├── BSR/                        # 图像分割测试数据集目录
├── results/                    # 分割、聚类结果与可视化输出目录
├── .idea/                      # PyCharm IDE配置文件
├── __pycache__/                # Python缓存目录
├── a.gitignore                 # Git忽略规则配置
└── README.md                   # 项目说明文档
运行步骤
打开 PyCharm，导入本项目文件夹
配置 Python 解释器，安装全部依赖
准备测试素材放入对应文件夹
直接运行 main.py，自动弹出桌面交互窗口
系统分为两大功能标签页：
图像分割模块：上传图片→选择算法→调整参数→一键运行，展示原图、分割图、轮廓系数
新闻聚类模块：上传 csv 新闻数据集→设置算法参数→执行聚类，输出评估指标 + PCA 可视化图
系统功能亮点
双算法对比：K-Means（指定簇数）、DBSCAN（密度自动分簇）
完整数据预处理流水线，适配图像、文本两类数据
课程指定评估指标：轮廓系数自动计算输出
独立桌面 Qt 界面，答辩演示直观，可打包 exe
模块化代码，算法与界面完全解耦，便于修改拓展
全程无需 Qt Designer 手动拖拽，代码一键运行
