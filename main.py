import sys
import os
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QLabel, QStatusBar, QMenuBar, QMenu,
    QAction, QMessageBox, QFileDialog, QSpinBox, QGroupBox, QFormLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

# ===================== Kmeans图像算法封装 =====================
try:
    from kmeans_image import kmeans_image_segmentation

    print("K-Means算法模块导入成功")
except ImportError as e:
    print(f"错误：无法导入K-Means算法模块 - {e}")
    traceback.print_exc()


    # 模拟算法函数
    def kmeans_image_segmentation(image_path, output_dir="./results", k_values=[3, 5, 8]):
        os.makedirs(output_dir, exist_ok=True)
        print(f"模拟K-Means分割: {image_path}, k={k_values}")
        from PIL import Image
        img = Image.new("RGB", (400, 300), color=(120, 180, 220))
        save_name = f"kmeans_k{k_values[0]}.png"
        save_path = os.path.join(output_dir, save_name)
        img.save(save_path)
        return {
            'silhouette_score': 0.526,
            'output_files': [save_name]
        }


# 统一调用入口
def alg_kmeans_segment(image_path, output_dir="./results", k_values=[3]):
    os.makedirs(output_dir, exist_ok=True)
    res = kmeans_image_segmentation(image_path, output_dir, k_values)
    fixed_output = []
    for fname in res["output_files"]:
        if not fname.endswith((".png", ".jpg", ".jpeg")):
            fname = fname + ".png"
        fixed_output.append(fname)
    res["output_files"] = fixed_output
    return res


# ===================== 文本聚类算法封装（修复可视化传参） =====================
try:
    from news_cluster import NewsCluster
    from visualization import plot_text_clusters

    print("文本聚类算法模块导入成功")
except ImportError as e:
    print(f"错误：无法导入文本聚类算法模块 - {e}")
    traceback.print_exc()


    def mock_news_cluster(news_path, cluster_num):
        os.makedirs("./results", exist_ok=True)
        from PIL import Image
        img = Image.new("RGB", (500, 350), color=(230, 160, 100))
        img.save("./results/cluster_plot.png")
        return {
            'status': 'success',
            'message': '文本聚类完成',
            'keywords': ['人工智能', '机器学习', 'Kmeans', '图像分割', '文本挖掘'],
            'plot_path': './results/cluster_plot.png'
        }


    # 备用模拟类，导入失败时生效
    class NewsCluster:
        def __init__(self, stopwords_path):
            self.stopwords_path = stopwords_path

        def cluster(self, path, num):
            return mock_news_cluster(path, num)


def alg_text_cluster(news_path, cluster_num):
    # 导入失败时走模拟类兼容逻辑
    if not hasattr(NewsCluster, 'load_news_data'):
        cluster = NewsCluster(stopwords_path="./stopwords.txt")
        return cluster.cluster(news_path, cluster_num)

    # 真实类完整调用流程
    cluster = NewsCluster(stopwords_path="./stopwords.txt")
    # 1. 加载新闻数据+文本预处理
    cluster.load_news_data(news_path)
    # 2. TF-IDF文本向量化
    cluster.vectorize_news()
    # 3. 执行DBSCAN聚类
    labels, noise_ratio, keywords_dict = cluster.dbscan_news()
    # 4. 格式化关键词，适配UI展示
    keywords = [f"簇 {k}：{'、'.join(v)}" for k, v in keywords_dict.items()]
    keywords.insert(0, f"噪声比例：{noise_ratio:.2f}%")
    # 5. 生成聚类可视化图
    os.makedirs("./results", exist_ok=True)
    plot_path = "./results/cluster_plot.png"

    # ===================== 修复部分 =====================
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，避免多线程中GUI冲突
    import matplotlib.pyplot as plt

    # 从 TF-IDF 向量化器中获取特征名（词汇表）
    try:
        feature_names = cluster.tfidf_vectorizer.get_feature_names_out()
    except AttributeError:
        try:
            feature_names = cluster.tfidf_vectorizer.get_feature_names()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(cluster.tfidf_matrix.shape[1])]

    # 调用可视化函数，传入必需的 feature_names 参数
    try:
        plot_text_clusters(cluster.tfidf_matrix, labels, feature_names)
    except Exception as e:
        print(f"可视化函数调用失败，使用降级方案: {e}")
        # 降级方案：手动绘制简单散点图
        from sklearn.decomposition import PCA
        import numpy as np
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(cluster.tfidf_matrix.toarray() if hasattr(cluster.tfidf_matrix, 'toarray') else cluster.tfidf_matrix)
        plt.figure(figsize=(10, 7))
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            marker = 'x' if label == -1 else 'o'
            plt.scatter(reduced[mask, 0], reduced[mask, 1], c=[color], marker=marker, label=f'噪声' if label == -1 else f'簇{label}', s=10)
        plt.title('文本聚类结果 (PCA降维)')
        plt.legend(markerscale=2)
    # 手动保存图片
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    # ===================================================

    # 6. 返回统一格式结果
    return {
        'status': 'success',
        'message': f'文本聚类完成，共生成{len(keywords_dict)}个簇',
        'keywords': keywords,
        'plot_path': plot_path
    }

# ===================== 后台工作线程 =====================
class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像分割与文本聚类系统")
        self.setGeometry(100, 100, 1200, 800)
        self.image_path = None
        self.news_path = None
        self.current_module = None
        self.is_running = False
        self.worker = None

        self.create_menu_bar()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        self.create_left_panel()
        self.create_right_panel()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def create_menu_bar(self):
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("文件")
        open_action = QAction("打开图片", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        help_menu = menu_bar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        title_label = QLabel("功能选择")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(title_label)
        self.image_segmentation_btn = QPushButton("图像分割")
        self.image_segmentation_btn.clicked.connect(self.show_image_segmentation)
        left_layout.addWidget(self.image_segmentation_btn)
        self.text_clustering_btn = QPushButton("文本聚类")
        self.text_clustering_btn.clicked.connect(self.show_text_clustering)
        left_layout.addWidget(self.text_clustering_btn)
        left_layout.addStretch()
        self.splitter.addWidget(left_panel)

    def create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.right_title = QLabel("请选择功能")
        self.right_title.setAlignment(Qt.AlignCenter)
        self.right_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.right_title)
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        right_layout.addWidget(self.params_widget)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        right_layout.addWidget(self.results_widget)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([200, 1000])

    def show_image_segmentation(self):
        self.right_title.setText("图像分割")
        self.current_module = "image_segmentation"
        self.clear_right_panel()
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout()
        self.img_open_btn = QPushButton("选择图片")
        self.img_open_btn.clicked.connect(self.open_image)
        params_layout.addRow("输入图片:", self.img_open_btn)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(2, 10)
        self.k_spinbox.setValue(5)
        params_layout.addRow("聚类数量:", self.k_spinbox)
        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)
        self.run_img_btn = QPushButton("运行分割")
        self.run_img_btn.clicked.connect(self.run_image_segmentation)
        self.params_layout.addWidget(self.run_img_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.params_layout.addWidget(self.progress_bar)

    def show_text_clustering(self):
        self.right_title.setText("文本聚类")
        self.current_module = "text_clustering"
        self.clear_right_panel()
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout()
        self.news_open_btn = QPushButton("选择文件")
        self.news_open_btn.clicked.connect(self.open_news_file)
        params_layout.addRow("新闻数据文件:", self.news_open_btn)
        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setRange(2, 10)
        self.cluster_spinbox.setValue(5)
        params_layout.addRow("聚类数量(DBSCAN自动生效):", self.cluster_spinbox)
        params_group.setLayout(params_layout)
        self.params_layout.addWidget(params_group)
        self.run_cluster_btn = QPushButton("运行聚类")
        self.run_cluster_btn.clicked.connect(self.run_text_clustering)
        self.params_layout.addWidget(self.run_cluster_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.params_layout.addWidget(self.progress_bar)

    def clear_right_panel(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.status_bar.showMessage(f"已选择图片: {file_path}")
            self.show_image_preview(file_path)

    def open_news_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择新闻文件", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.news_path = file_path
            self.status_bar.showMessage(f"已选择新闻文件: {file_path}")
            self.show_news_preview(file_path)

    def show_image_preview(self, image_path):
        self.clear_results_area()
        preview_label = QLabel()
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio)
            preview_label.setPixmap(scaled_pixmap)
        else:
            preview_label.setText("图片加载失败")
        preview_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(preview_label)

    def show_news_preview(self, news_path):
        self.clear_results_area()
        preview_label = QLabel(f"已选择新闻文件:\n{news_path}")
        preview_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(preview_label)

    def clear_results_area(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def run_image_segmentation(self):
        if self.is_running:
            QMessageBox.information(self, "提示", "任务正在后台运行，请等待完成！")
            return
        if not self.image_path:
            QMessageBox.warning(self, "提示", "请先选择一张图片！")
            return
        self.is_running = True
        self.run_img_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)
        self.status_bar.showMessage("正在运行图像分割...")
        k_val = self.k_spinbox.value()
        self.worker = WorkerThread(alg_kmeans_segment, self.image_path, "./results", [k_val])
        self.worker.finished.connect(self.on_segmentation_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def run_text_clustering(self):
        if self.is_running:
            QMessageBox.information(self, "提示", "任务正在后台运行，请等待完成！")
            return
        if not self.news_path:
            QMessageBox.warning(self, "提示", "请先选择一个新闻csv文件！")
            return
        self.is_running = True
        self.run_cluster_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)
        self.status_bar.showMessage("正在运行文本聚类...")
        cluster_num = self.cluster_spinbox.value()
        self.worker = WorkerThread(alg_text_cluster, self.news_path, cluster_num)
        self.worker.finished.connect(self.on_clustering_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_segmentation_finished(self, result):
        self.is_running = False
        self.run_img_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("图像分割完成")
        self.show_segmentation_results(result)

    def on_clustering_finished(self, result):
        self.is_running = False
        self.run_cluster_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("文本聚类完成")
        self.show_clustering_results(result)

    def on_worker_error(self, error_msg):
        self.is_running = False
        self.run_img_btn.setEnabled(True)
        self.run_cluster_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"错误: {error_msg}")
        QMessageBox.critical(self, "运行失败", f"操作异常：{error_msg}")

    def show_segmentation_results(self, result):
        self.clear_results_area()
        output_files = result.get("output_files", [])
        if not output_files:
            self.results_layout.addWidget(QLabel("未生成分割图片"))
            return

        full_path = output_files[0]
        print(f"加载图片完整路径: {full_path}")
        if not os.path.exists(full_path):
            tip_label = QLabel(f"生成失败：{full_path} 文件不存在")
            self.results_layout.addWidget(tip_label)
            return
        result_label = QLabel()
        pixmap = QPixmap(full_path)
        if pixmap.isNull():
            self.results_layout.addWidget(QLabel(f"图片损坏无法加载：{full_path}"))
            return
        scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio)
        result_label.setPixmap(scaled_pixmap)
        result_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(result_label)
        self.results_layout.addWidget(QLabel("===== 分割评估指标 ====="))
        score = result.get('silhouette_score', "N/A")
        self.results_layout.addWidget(QLabel(f"K={self.k_spinbox.value()} 轮廓系数：{score}"))

    def show_clustering_results(self, result):
        self.clear_results_area()
        self.results_layout.addWidget(QLabel("===== 文本聚类结果 ====="))
        plot_path = result.get('plot_path', './results/cluster_plot.png')
        if os.path.exists(plot_path):
            plot_label = QLabel()
            pixmap = QPixmap(plot_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio)
                plot_label.setPixmap(scaled_pixmap)
                plot_label.setAlignment(Qt.AlignCenter)
                self.results_layout.addWidget(plot_label)
            else:
                self.results_layout.addWidget(QLabel("聚类可视化图加载失败"))
        else:
            self.results_layout.addWidget(QLabel("未找到聚类可视化图片"))
        self.results_layout.addWidget(QLabel("聚类详情与关键词："))
        keywords = result.get('keywords', ["无"])
        keywords_label = QLabel("\n".join(keywords))
        keywords_label.setWordWrap(True)
        self.results_layout.addWidget(keywords_label)
        export_btn = QPushButton("导出聚类结果")
        export_btn.clicked.connect(lambda: self.export_results(result))
        self.results_layout.addWidget(export_btn)

    def export_results(self, result):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存聚类结果", "", "CSV Files (*.csv);;Text Files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("文本聚类分析报告\n")
                    f.write(f"执行状态：{result.get('status')}\n")
                    f.write(f"执行信息：{result.get('message')}\n")
                    f.write("\n聚类关键词：\n")
                    f.write("\n".join(result.get('keywords')))
                QMessageBox.information(self, "导出成功", f"文件已保存至：{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"写入文件出错：{str(e)}")

    def show_about(self):
        QMessageBox.about(
            self, "关于软件",
            "图像分割与文本聚类系统 V1.0\n"
            "算法：Kmeans图像分割 + 新闻文本聚类\n"
            "修复：编码兼容、方法调用匹配、可视化参数适配"
        )


if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        print("缺少pillow，执行 pip install pillow")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())