import sys
import os
import traceback
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QLabel, QStatusBar, QMenuBar, QAction,
    QMessageBox, QFileDialog, QSpinBox, QGroupBox, QFormLayout,
    QProgressBar, QComboBox, QScrollArea, QFrame, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from preprocessing import DataPreprocessor
from kmeans_image import kmeans_image_segmentation as _kmeans_img_seg
from news_cluster import NewsCluster
from visualization import plot_text_clusters as _plot_text_clusters


def alg_kmeans_image(image_path, output_dir="./results", k=5):
    os.makedirs(output_dir, exist_ok=True)
    try:
        res = _kmeans_img_seg(image_path, output_dir, [k])
    except Exception as e:
        return {'algorithm': 'K-Means', 'module': 'image', 'error': str(e)}

    if not res or 'output_files' not in res or not res['output_files']:
        return {'algorithm': 'K-Means', 'module': 'image', 'error': 'K-Means 分割未生成结果'}

    output_files = res['output_files']
    fixed_output = []
    for fname in output_files:
        if not fname.endswith((".png", ".jpg", ".jpeg")):
            fname = fname + ".png"
        fixed_output.append(fname)

    score = res.get('silhouette_score', None)
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
    return {
        'algorithm': 'K-Means', 'module': 'image', 'image_path': fixed_output[0],
        'metrics': {'K值': k, '轮廓系数': score_str}
    }


def alg_dbscan_image(image_path, output_dir="./results", eps=0.15, min_samples=30, spatial_weight=0.1):
    """DBSCAN 图像分割 — 修复版：加入空间坐标特征，使用簇平均颜色"""
    os.makedirs(output_dir, exist_ok=True)
    try:
        processor = DataPreprocessor("./stopwords.txt")
        img = processor.read_image(image_path)
    except Exception as e:
        return {'algorithm': 'DBSCAN', 'module': 'image', 'error': f'图像读取失败: {e}'}

    if img is None:
        return {'algorithm': 'DBSCAN', 'module': 'image', 'error': '无法读取图像'}

    h, w, c = img.shape

    # ================= 核心修复 1：加入空间坐标 =================
    ys, xs = np.mgrid[0:h, 0:w]
    xs_n = (xs / w).astype(np.float32) * spatial_weight  # 归一化并赋予较小权重
    ys_n = (ys / h).astype(np.float32) * spatial_weight

    # 5维特征：R, G, B, x, y
    features = np.dstack([img, xs_n, ys_n]).reshape(-1, 5)

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(features)
    except Exception as e:
        return {'algorithm': 'DBSCAN', 'module': 'image', 'error': f'DBSCAN 运行失败: {e}'}

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / len(labels) * 100

    if n_clusters == 0:
        return {'algorithm': 'DBSCAN', 'module': 'image',
                'error': f'所有像素都被标记为噪声(噪声比例{noise_ratio:.1f}%)，请增大 eps 或减小 min_samples'}

    # ================= 核心修复 2：使用簇平均颜色而非随机色 =================
    flat_img = img.reshape(-1, 3)
    segmented_img = np.zeros((h * w, 3), dtype=np.uint8)

    for label in unique_labels:
        if label == -1:
            segmented_img[labels == label] = [0, 0, 0]  # 噪声为黑色
        else:
            # 用该簇所有像素的平均颜色填充
            mean_color = flat_img[labels == label].mean(axis=0)
            segmented_img[labels == label] = mean_color

    segmented_img = segmented_img.reshape((h, w, 3))

    metrics = {'聚类数': n_clusters, '噪声比例': f"{noise_ratio:.2f}%"}
    if n_clusters > 1:
        valid_mask = labels != -1
        if valid_mask.sum() > n_clusters:
            try:
                score = silhouette_score(features[valid_mask], labels[valid_mask])
                metrics['轮廓系数'] = f"{score:.4f}"
            except Exception:
                metrics['轮廓系数'] = 'N/A'

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"dbscan_{img_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

    return {'algorithm': 'DBSCAN', 'module': 'image', 'image_path': save_path, 'metrics': metrics}


def alg_kmeans_text(news_path, k=5):
    output_dir = "./results/kmeans_text"
    os.makedirs(output_dir, exist_ok=True)
    try:
        cluster = NewsCluster(stopwords_path="./stopwords.txt")
        cluster.load_news_data(news_path)
        cluster.vectorize_news()
    except Exception as e:
        return {'algorithm': 'K-Means', 'module': 'text', 'error': f'数据加载失败: {e}'}

    tfidf = cluster.tfidf_matrix
    feature_names = cluster.feature_names
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf)
    except Exception as e:
        return {'algorithm': 'K-Means', 'module': 'text', 'error': f'K-Means 聚类失败: {e}'}

    try:
        score = silhouette_score(tfidf, labels)
        score_str = f"{score:.4f}"
    except Exception:
        score_str = "N/A"

    keywords_dict = cluster.get_top_keywords(tfidf, feature_names, labels)
    try:
        tfidf_dense = tfidf.toarray()
        _plot_text_clusters(tfidf_dense, labels, list(feature_names), output_dir=output_dir)
    except Exception as e:
        print(f"可视化失败: {e}")

    plot_path = os.path.join(output_dir, "text_clusters.png")
    keywords_list = [f"簇 {cid}：{'、'.join(kw)}" for cid, kw in keywords_dict.items()]
    keywords_list.insert(0, f"轮廓系数：{score_str}")
    return {'algorithm': 'K-Means', 'module': 'text', 'plot_path': plot_path,
            'metrics': {'K值': k, '轮廓系数': score_str}, 'keywords': keywords_list}


def alg_dbscan_text(news_path, eps=0.5, min_samples=10):
    output_dir = "./results/dbscan_text"
    os.makedirs(output_dir, exist_ok=True)
    try:
        cluster = NewsCluster(stopwords_path="./stopwords.txt")
        cluster.load_news_data(news_path)
        cluster.vectorize_news()
    except Exception as e:
        return {'algorithm': 'DBSCAN', 'module': 'text', 'error': f'数据加载失败: {e}'}

    try:
        labels, noise_ratio, keywords_dict = cluster.dbscan_news(eps=eps, min_samples=min_samples)
    except Exception as e:
        return {'algorithm': 'DBSCAN', 'module': 'text', 'error': f'DBSCAN 聚类失败: {e}'}

    try:
        tfidf_dense = cluster.tfidf_matrix.toarray()
        _plot_text_clusters(tfidf_dense, labels, list(cluster.feature_names), output_dir=output_dir)
    except Exception as e:
        print(f"可视化失败: {e}")

    plot_path = os.path.join(output_dir, "text_clusters.png")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    keywords_list = [f"簇 {cid}：{'、'.join(kw)}" for cid, kw in keywords_dict.items()]
    keywords_list.insert(0, f"噪声比例：{noise_ratio:.2f}%")
    return {'algorithm': 'DBSCAN', 'module': 'text', 'plot_path': plot_path,
            'metrics': {'聚类数': n_clusters, '噪声比例': f"{noise_ratio:.2f}%", 'eps': eps,
                        'min_samples': min_samples}, 'keywords': keywords_list}


class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

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
        self.setWindowTitle("图像分割与文本聚类系统（K-Means + DBSCAN 双算法）")
        self.setGeometry(100, 100, 1400, 900)
        self.image_path = None
        self.news_path = None
        self.current_module = None
        self.is_running = False
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        self._create_menu_bar()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        self._create_left_panel()
        self._create_right_panel()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 请选择功能")

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        open_image_action = QAction("打开图片", self)
        open_image_action.triggered.connect(self._open_image)
        file_menu.addAction(open_image_action)
        open_news_action = QAction("打开新闻CSV", self)
        open_news_action.triggered.connect(self._open_news_file)
        file_menu.addAction(open_news_action)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        help_menu = menu_bar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("功能选择")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        left_layout.addWidget(title)

        self.btn_image = QPushButton("图像分割")
        self.btn_image.setStyleSheet(
            "font-size: 18px; padding: 24px; border-radius: 8px; "
            "background-color: #4CAF50; color: white;"
        )
        self.btn_image.clicked.connect(self._show_image_segmentation)
        left_layout.addWidget(self.btn_image)

        self.btn_text = QPushButton("文本聚类")
        self.btn_text.setStyleSheet(
            "font-size: 18px; padding: 24px; border-radius: 8px; "
            "background-color: #2196F3; color: white;"
        )
        self.btn_text.clicked.connect(self._show_text_clustering)
        left_layout.addWidget(self.btn_text)

        left_layout.addStretch()

        info_label = QLabel(
            "支持算法：\n"
            "• K-Means（需指定K值）\n"
            "• DBSCAN（密度聚类）\n\n"
            "可依次运行两种算法\n"
            "结果并列展示，互不覆盖"
        )
        info_label.setStyleSheet("color: gray; font-size: 13px; padding: 10px;")
        info_label.setWordWrap(True)
        left_layout.addWidget(info_label)
        self.splitter.addWidget(left_panel)

    def _create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.right_title = QLabel("请选择功能")
        self.right_title.setAlignment(Qt.AlignCenter)
        self.right_title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        right_layout.addWidget(self.right_title)

        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        right_layout.addWidget(self.params_widget)

        results_label = QLabel("运行结果（累积展示，互不覆盖）")
        results_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_layout.addWidget(results_label)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setStyleSheet("QScrollArea { border: 1px solid #ccc; border-radius: 5px; }")
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_scroll.setWidget(self.results_container)
        right_layout.addWidget(self.results_scroll, 1)

        self.clear_btn = QPushButton("🗑 清空所有结果")
        self.clear_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        self.clear_btn.clicked.connect(self._clear_all_results)
        right_layout.addWidget(self.clear_btn)

        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([250, 1150])

    def _show_image_segmentation(self):
        self.right_title.setText("📷 图像分割（K-Means / DBSCAN）")
        self.current_module = "image"
        self._clear_params()

        file_group = QGroupBox("输入文件")
        file_layout = QFormLayout()
        self.img_open_btn = QPushButton("选择图片")
        self.img_open_btn.clicked.connect(self._open_image)
        file_layout.addRow("图片路径:", self.img_open_btn)
        file_group.setLayout(file_layout)
        self.params_layout.addWidget(file_group)

        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout()
        self.img_algo_combo = QComboBox()
        self.img_algo_combo.addItems(["K-Means", "DBSCAN"])
        self.img_algo_combo.currentIndexChanged.connect(self._on_img_algo_changed)
        algo_layout.addWidget(self.img_algo_combo)
        algo_group.setLayout(algo_layout)
        self.params_layout.addWidget(algo_group)

        self.kmeans_img_group = QGroupBox("K-Means 参数")
        km_layout = QFormLayout()
        self.img_k_spinbox = QSpinBox()
        self.img_k_spinbox.setRange(2, 10)
        self.img_k_spinbox.setValue(5)
        km_layout.addRow("聚类数 K:", self.img_k_spinbox)
        self.kmeans_img_group.setLayout(km_layout)
        self.params_layout.addWidget(self.kmeans_img_group)

        self.dbscan_img_group = QGroupBox("DBSCAN 参数")
        db_layout = QFormLayout()
        self.img_eps_spinbox = QDoubleSpinBox()
        self.img_eps_spinbox.setRange(0.01, 0.50)
        self.img_eps_spinbox.setSingleStep(0.01)
        self.img_eps_spinbox.setValue(0.15)  # 默认值修改
        self.img_eps_spinbox.setDecimals(2)
        db_layout.addRow("eps (邻域半径):", self.img_eps_spinbox)

        self.img_min_samples_spinbox = QSpinBox()
        self.img_min_samples_spinbox.setRange(5, 500)
        self.img_min_samples_spinbox.setValue(30)  # 默认值修改
        self.img_min_samples_spinbox.setSingleStep(5)
        db_layout.addRow("min_samples (最小点数):", self.img_min_samples_spinbox)
        self.dbscan_img_group.setLayout(db_layout)
        self.params_layout.addWidget(self.dbscan_img_group)

        self.run_img_btn = QPushButton("▶ 运行图像分割")
        self.run_img_btn.setStyleSheet(
            "font-size: 16px; padding: 12px; border-radius: 5px; "
            "background-color: #4CAF50; color: white;"
        )
        self.run_img_btn.clicked.connect(self._run_image_segmentation)
        self.params_layout.addWidget(self.run_img_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.params_layout.addWidget(self.progress_bar)
        self._on_img_algo_changed(0)

    def _on_img_algo_changed(self, index):
        is_kmeans = (self.img_algo_combo.currentText() == "K-Means")
        self.kmeans_img_group.setVisible(is_kmeans)
        self.dbscan_img_group.setVisible(not is_kmeans)

    def _show_text_clustering(self):
        self.right_title.setText("📝 文本聚类（K-Means / DBSCAN）")
        self.current_module = "text"
        self._clear_params()

        file_group = QGroupBox("输入文件")
        file_layout = QFormLayout()
        self.news_open_btn = QPushButton("选择CSV文件")
        self.news_open_btn.clicked.connect(self._open_news_file)
        file_layout.addRow("新闻数据:", self.news_open_btn)
        file_group.setLayout(file_layout)
        self.params_layout.addWidget(file_group)

        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout()
        self.text_algo_combo = QComboBox()
        self.text_algo_combo.addItems(["K-Means", "DBSCAN"])
        self.text_algo_combo.currentIndexChanged.connect(self._on_text_algo_changed)
        algo_layout.addWidget(self.text_algo_combo)
        algo_group.setLayout(algo_layout)
        self.params_layout.addWidget(algo_group)

        self.kmeans_text_group = QGroupBox("K-Means 参数")
        km_layout = QFormLayout()
        self.text_k_spinbox = QSpinBox()
        self.text_k_spinbox.setRange(2, 10)
        self.text_k_spinbox.setValue(5)
        km_layout.addRow("聚类数 K:", self.text_k_spinbox)
        self.kmeans_text_group.setLayout(km_layout)
        self.params_layout.addWidget(self.kmeans_text_group)

        # 新增：DBSCAN参数调整组
        self.dbscan_text_group = QGroupBox("DBSCAN 参数")
        db_layout = QFormLayout()

        self.text_eps_spinbox = QDoubleSpinBox()
        self.text_eps_spinbox.setRange(0.1, 1.0)
        self.text_eps_spinbox.setSingleStep(0.1)
        self.text_eps_spinbox.setValue(0.5)  # 默认值修改
        self.text_eps_spinbox.setDecimals(2)
        db_layout.addRow("eps (邻域半径):", self.text_eps_spinbox)

        self.text_min_samples_spinbox = QSpinBox()
        self.text_min_samples_spinbox.setRange(5, 50)
        self.text_min_samples_spinbox.setValue(10)  # 默认值修改
        self.text_min_samples_spinbox.setSingleStep(5)
        db_layout.addRow("min_samples (最小点数):", self.text_min_samples_spinbox)

        self.dbscan_text_group.setLayout(db_layout)
        self.params_layout.addWidget(self.dbscan_text_group)

        self.run_text_btn = QPushButton("▶ 运行文本聚类")
        self.run_text_btn.setStyleSheet(
            "font-size: 16px; padding: 12px; border-radius: 5px; "
            "background-color: #2196F3; color: white;"
        )
        self.run_text_btn.clicked.connect(self._run_text_clustering)
        self.params_layout.addWidget(self.run_text_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.params_layout.addWidget(self.progress_bar)
        self._on_text_algo_changed(0)

    def _run_text_clustering(self):
        if self.is_running:
            QMessageBox.information(self, "提示", "任务正在运行中，请等待完成！")
            return
        if not self.news_path:
            QMessageBox.warning(self, "提示", "请先选择一个新闻CSV文件！")
            return

        self.is_running = True
        self.run_text_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)

        algo = self.text_algo_combo.currentText()
        if algo == "K-Means":
            k = self.text_k_spinbox.value()
            self.status_bar.showMessage(f"正在运行 K-Means 文本聚类 (K={k})...")
            self.worker = WorkerThread(alg_kmeans_text, self.news_path, k)
        else:
            eps = self.text_eps_spinbox.value()
            min_samples = self.text_min_samples_spinbox.value()
            self.status_bar.showMessage(f"正在运行 DBSCAN 文本聚类 (eps={eps}, min_samples={min_samples})...")
            self.worker = WorkerThread(alg_dbscan_text, self.news_path, eps, min_samples)

        self.worker.finished.connect(self._on_text_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_text_algo_changed(self, index):
        is_kmeans = (self.text_algo_combo.currentText() == "K-Means")
        self.kmeans_text_group.setVisible(is_kmeans)
        self.dbscan_text_group.setVisible(not is_kmeans)

    def _open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.status_bar.showMessage(f"已选择图片: {file_path}")

    def _open_news_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择新闻CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.news_path = file_path
            self.status_bar.showMessage(f"已选择新闻文件: {file_path}")

    def _run_image_segmentation(self):
        if self.is_running:
            QMessageBox.information(self, "提示", "任务正在运行中，请等待完成！")
            return
        if not self.image_path:
            QMessageBox.warning(self, "提示", "请先选择一张图片！")
            return

        self.is_running = True
        self.run_img_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)

        algo = self.img_algo_combo.currentText()
        if algo == "K-Means":
            k = self.img_k_spinbox.value()
            self.status_bar.showMessage(f"正在运行 K-Means 图像分割 (K={k})...")
            self.worker = WorkerThread(alg_kmeans_image, self.image_path, "./results", k)
        else:
            eps = self.img_eps_spinbox.value()
            min_samples = self.img_min_samples_spinbox.value()
            self.status_bar.showMessage(f"正在运行 DBSCAN 图像分割 (eps={eps}, min_samples={min_samples})...")
            self.worker = WorkerThread(alg_dbscan_image, self.image_path, "./results", eps, min_samples)

        self.worker.finished.connect(self._on_image_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _run_text_clustering(self):
        if self.is_running:
            QMessageBox.information(self, "提示", "任务正在运行中，请等待完成！")
            return
        if not self.news_path:
            QMessageBox.warning(self, "提示", "请先选择一个新闻CSV文件！")
            return

        self.is_running = True
        self.run_text_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)

        algo = self.text_algo_combo.currentText()
        if algo == "K-Means":
            k = self.text_k_spinbox.value()
            self.status_bar.showMessage(f"正在运行 K-Means 文本聚类 (K={k})...")
            self.worker = WorkerThread(alg_kmeans_text, self.news_path, k)
        else:
            self.status_bar.showMessage("正在运行 DBSCAN 文本聚类（自动搜索参数）...")
            self.worker = WorkerThread(alg_dbscan_text, self.news_path)

        self.worker.finished.connect(self._on_text_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_image_finished(self, result):
        self.is_running = False
        self.run_img_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        algo = result.get('algorithm', 'Unknown')
        self.status_bar.showMessage(f"{algo} 图像分割完成")
        self._add_result_widget(result)

    def _on_text_finished(self, result):
        self.is_running = False
        self.run_text_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        algo = result.get('algorithm', 'Unknown')
        self.status_bar.showMessage(f"{algo} 文本聚类完成")
        self._add_result_widget(result)

    def _on_worker_error(self, error_msg):
        self.is_running = False
        if hasattr(self, 'run_img_btn'): self.run_img_btn.setEnabled(True)
        if hasattr(self, 'run_text_btn'): self.run_text_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"错误: {error_msg}")
        QMessageBox.critical(self, "运行失败", f"操作异常：{error_msg}")

    def _add_result_widget(self, result):
        algo = result.get('algorithm', 'Unknown')
        module = result.get('module', 'Unknown')

        if result.get('error'):
            error_group = QGroupBox(f"❌ {algo} {'图像分割' if module == 'image' else '文本聚类'} — 运行失败")
            error_layout = QVBoxLayout()
            error_label = QLabel(result['error'])
            error_label.setWordWrap(True)
            error_label.setStyleSheet("color: red; font-size: 14px;")
            error_layout.addWidget(error_label)
            error_group.setLayout(error_layout)
            self.results_layout.addWidget(error_group)
            return

        module_text = "图像分割" if module == 'image' else "文本聚类"
        result_group = QGroupBox(f"✅ {algo} {module_text} 结果")
        result_group.setStyleSheet(
            "QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #ddd; border-radius: 5px; margin-top: 10px; padding-top: 15px; }"
        )
        result_layout = QVBoxLayout()

        image_path = result.get('image_path') or result.get('plot_path')
        if image_path and os.path.exists(image_path):
            img_label = QLabel()
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(500, 400, Qt.KeepAspectRatio)
                img_label.setPixmap(scaled_pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                result_layout.addWidget(img_label)
            else:
                result_layout.addWidget(QLabel("⚠ 图片加载失败"))
        else:
            result_layout.addWidget(QLabel("⚠ 未找到结果图片"))

        metrics = result.get('metrics', {})
        if metrics:
            metrics_title = QLabel("📊 评估指标")
            metrics_title.setStyleSheet("font-size: 16px; font-weight: bold; padding-top: 8px;")
            result_layout.addWidget(metrics_title)
            metrics_text = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
            metrics_label = QLabel(metrics_text)
            metrics_label.setStyleSheet("font-size: 14px; padding: 3px;")
            result_layout.addWidget(metrics_label)

        keywords = result.get('keywords', [])
        if keywords:
            kw_title = QLabel("🔑 聚类关键词")
            kw_title.setStyleSheet("font-size: 16px; font-weight: bold; padding-top: 8px;")
            result_layout.addWidget(kw_title)
            for kw in keywords:
                kw_label = QLabel(f"  {kw}")
                kw_label.setWordWrap(True)
                kw_label.setStyleSheet("font-size: 14px; padding: 2px;")
                result_layout.addWidget(kw_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        result_layout.addWidget(line)
        result_group.setLayout(result_layout)
        self.results_layout.addWidget(result_group)
        self.results_scroll.verticalScrollBar().setValue(self.results_scroll.verticalScrollBar().maximum())

    def _clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

    def _clear_all_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        self.status_bar.showMessage("已清空所有结果")

    def _show_about(self):
        QMessageBox.about(self, "关于软件",
                          "图像分割与文本聚类系统 V2.0\n\n支持算法：\n 📷 图像分割：K-Means + DBSCAN\n 📝 文本聚类：K-Means + DBSCAN")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 修改：全局放大字体并美化样式
    app.setStyleSheet("""
    QMainWindow { background-color: #f5f5f5; }
    QWidget { font-size: 15px; } /* 全局基础字体放大 */

    QMenuBar, QMenuBar::item { font-size: 16px; padding: 6px 14px; }
    QMenu::item { font-size: 15px; padding: 6px 24px; }

    QGroupBox {
        font-size: 16px; font-weight: bold;
        border: 2px solid #ccc; border-radius: 8px;
        margin-top: 14px; padding-top: 18px;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }

    QPushButton { font-size: 15px; padding: 10px 16px; border-radius: 6px; }
    QPushButton:hover { background-color: #45a049; }

    QSpinBox, QDoubleSpinBox, QComboBox {
        font-size: 15px; padding: 4px 8px; min-height: 24px;
    }

    QStatusBar { font-size: 14px; }
    QScrollArea { border: 1px solid #ccc; border-radius: 6px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
