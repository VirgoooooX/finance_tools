"""
财务数据分析工具 - PyQt5 GUI主程序
"""
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QProgressBar, QGroupBox, QCheckBox, QDoubleSpinBox, QMessageBox,
    QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from config_manager import ConfigManager


class ProcessThread(QThread):
    """后台处理线程"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str)  # log message
    
    def __init__(self, directory, config):
        super().__init__()
        self.directory = directory
        self.config = config
    
    def run(self):
        """运行数据处理"""
        try:
            # 动态导入避免循环依赖
            import financial_analyzer
            
            self.log.emit("🔄 开始处理...")
            self.progress.emit(0, 100, "加载分析模块...")
            
            # 实例化分析器并传入回调
            analyzer = financial_analyzer.FinancialAnalyzer(
                config=self.config, 
                progress_callback=self.progress_callback
            )
            
            # 开始执行处理
            analyzer.process_directory(self.directory)
            
            self.finished.emit(True, "数据处理完成")
        except Exception as e:
            self.log.emit(f"❌ 错误: {str(e)}")
            self.finished.emit(False, str(e))
    
    def progress_callback(self, current, total, message):
        """进度回调"""
        self.progress.emit(current, total, message)
        self.log.emit(message)


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.process_thread = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("财务数据分析工具 v2.0")
        self.setGeometry(100, 100, 1000, 750)
        
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QWidget {
                font-family: "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #e1e8ed;
                border-radius: 8px;
                margin-top: 12px;
                padding: 15px;
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #3498db;
            }
            QLineEdit {
                padding: 8px 12px;
                border: 2px solid #e1e8ed;
                border-radius: 6px;
                background-color: white;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
            QPushButton {
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                background-color: #ecf0f1;
                color: #2c3e50;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
            QPushButton:pressed {
                background-color: #95a5a6;
            }
            QCheckBox {
                spacing: 8px;
                color: #2c3e50;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #bdc3c7;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border-color: #3498db;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgNEw0LjUgNy41TDExIDEiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+);
            }
            QTextEdit {
                border: 1px solid #e1e8ed;
                border-radius: 6px;
                background-color: #fafbfc;
                padding: 8px;
                color: #2c3e50;
            }
            QProgressBar {
                border: none;
                border-radius: 8px;
                background-color: #ecf0f1;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2ecc71);
            }
            QLabel {
                color: #2c3e50;
            }
        """)
        
        # 主widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(main_layout)
        
        # 标题栏
        title_widget = QWidget()
        title_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 10px;
            padding: 20px;
        """)
        title_layout = QVBoxLayout(title_widget)
        
        title = QLabel("📊 财务数据分析工具")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        subtitle = QLabel("智能财务报表处理与分析系统")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-size: 11pt;")
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)
        
        main_layout.addWidget(title_widget)
        
        # 文件选择区域
        file_group = self.create_file_selection_group()
        main_layout.addWidget(file_group)
        
        # 配置选项区域
        config_group = self.create_config_group()
        main_layout.addWidget(config_group)
        
        # 操作按钮
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: white;
            padding: 10px;
            border-radius: 6px;
            color: #27ae60;
            font-weight: bold;
        """)
        main_layout.addWidget(self.status_label)
        
        # 日志区域
        log_container = QWidget()
        log_container.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            padding: 10px;
        """)
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_label = QLabel("📋 处理日志")
        log_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 11pt;")
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_container)
        
        # 加载配置到UI
        self.load_config_to_ui()
    
    def create_file_selection_group(self):
        """创建文件选择组"""
        group = QGroupBox("📁 数据源")
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        label = QLabel("工作目录:")
        label.setStyleSheet("font-weight: normal; min-width: 70px;")
        layout.addWidget(label)
        
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("选择包含Excel文件的目录...")
        self.dir_input.setText(os.getcwd())
        layout.addWidget(self.dir_input)
        
        browse_btn = QPushButton("📂 浏览")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        browse_btn.clicked.connect(self.browse_directory)
        layout.addWidget(browse_btn)
        
        group.setLayout(layout)
        return group
    
    
    def create_config_group(self):
        """创建配置选项组"""
        group = QGroupBox("⚙️ 配置选项")
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        
        # 输出选项
        output_container = QWidget()
        output_container.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 10px;")
        output_layout = QHBoxLayout(output_container)
        
        output_label = QLabel("输出内容:")
        output_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        output_layout.addWidget(output_label)
        
        self.cb_original = QCheckBox("原始数据")
        self.cb_original.setChecked(True)
        output_layout.addWidget(self.cb_original)
        
        self.cb_validation = QCheckBox("验证报告")
        self.cb_validation.setChecked(True)
        output_layout.addWidget(self.cb_validation)
        
        self.cb_metrics = QCheckBox("财务指标")
        self.cb_metrics.setChecked(True)
        output_layout.addWidget(self.cb_metrics)
        
        output_layout.addStretch()
        main_layout.addWidget(output_container)
        
        # 验证选项
        validation_container = QWidget()
        validation_container.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 10px;")
        validation_layout = QHBoxLayout(validation_container)
        
        self.cb_enable_validation = QCheckBox("启用会计恒等式验证")
        self.cb_enable_validation.setChecked(True)
        validation_layout.addWidget(self.cb_enable_validation)
        
        validation_layout.addWidget(QLabel("容差:"))
        self.tolerance_spinbox = QDoubleSpinBox()
        self.tolerance_spinbox.setRange(0, 1000)
        self.tolerance_spinbox.setValue(0.01)
        self.tolerance_spinbox.setDecimals(2)
        self.tolerance_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                padding: 6px;
                border: 2px solid #e1e8ed;
                border-radius: 4px;
                background-color: white;
                min-width: 80px;
            }
        """)
        validation_layout.addWidget(self.tolerance_spinbox)
        
        validation_layout.addStretch()
        main_layout.addWidget(validation_container)
        
        # 指标选项
        metrics_container = QWidget()
        metrics_container.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 10px;")
        metrics_layout = QHBoxLayout(metrics_container)
        
        metrics_label = QLabel("计算指标:")
        metrics_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        metrics_layout.addWidget(metrics_label)
        
        self.cb_liquidity = QCheckBox("流动性")
        self.cb_liquidity.setChecked(True)
        metrics_layout.addWidget(self.cb_liquidity)
        
        self.cb_solvency = QCheckBox("偿债能力")
        self.cb_solvency.setChecked(True)
        metrics_layout.addWidget(self.cb_solvency)
        
        self.cb_profitability = QCheckBox("盈利能力")
        self.cb_profitability.setChecked(True)
        metrics_layout.addWidget(self.cb_profitability)
        
        self.cb_cashflow = QCheckBox("现金流")
        self.cb_cashflow.setChecked(True)
        metrics_layout.addWidget(self.cb_cashflow)
        
        metrics_layout.addStretch()
        main_layout.addWidget(metrics_container)
        
        group.setLayout(main_layout)
        return group
    
    def create_button_layout(self):
        """创建按钮布局"""
        layout = QHBoxLayout()
        layout.setSpacing(12)
        
        self.edit_mapping_btn = QPushButton("📝 编辑科目映射")
        self.edit_mapping_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 12px 25px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.edit_mapping_btn.clicked.connect(self.edit_account_mapping)
        layout.addWidget(self.edit_mapping_btn)
        
        self.reset_btn = QPushButton("🔄 重置配置")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 12px 25px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_to_default)
        layout.addWidget(self.reset_btn)
        
        layout.addStretch()
        
        self.start_btn = QPushButton("▶️ 开始处理")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:1 #27ae60);
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 15px 50px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #229954);
            }
            QPushButton:pressed {
                background: #1e8449;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)
        
        return layout
    
    def load_config_to_ui(self):
        """从配置加载到UI"""
        config = self.config_manager.config
        
        # 输出选项
        output_opts = config.get('输出选项', {})
        self.cb_original.setChecked(output_opts.get('生成原始数据', True))
        self.cb_validation.setChecked(output_opts.get('生成验证报告', True))
        self.cb_metrics.setChecked(output_opts.get('生成财务指标', True))
        
        # 验证选项
        val_opts = config.get('验证选项', {})
        self.cb_enable_validation.setChecked(val_opts.get('启用会计恒等式验证', True))
        self.tolerance_spinbox.setValue(val_opts.get('容差阈值', 0.01))
        
        # 指标选项
        metric_opts = config.get('指标选项', {})
        self.cb_liquidity.setChecked(metric_opts.get('计算流动性指标', True))
        self.cb_solvency.setChecked(metric_opts.get('计算偿债能力指标', True))
        self.cb_profitability.setChecked(metric_opts.get('计算盈利能力指标', True))
        self.cb_cashflow.setChecked(metric_opts.get('计算现金流指标', True))
    
    def save_ui_to_config(self):
        """从UI保存到配置"""
        # 输出选项
        self.config_manager.set('输出选项.生成原始数据', self.cb_original.isChecked())
        self.config_manager.set('输出选项.生成验证报告', self.cb_validation.isChecked())
        self.config_manager.set('输出选项.生成财务指标', self.cb_metrics.isChecked())
        
        # 验证选项
        self.config_manager.set('验证选项.启用会计恒等式验证', self.cb_enable_validation.isChecked())
        self.config_manager.set('验证选项.容差阈值', self.tolerance_spinbox.value())
        
        # 指标选项
        self.config_manager.set('指标选项.计算流动性指标', self.cb_liquidity.isChecked())
        self.config_manager.set('指标选项.计算偿债能力指标', self.cb_solvency.isChecked())
        self.config_manager.set('指标选项.计算盈利能力指标', self.cb_profitability.isChecked())
        self.config_manager.set('指标选项.计算现金流指标', self.cb_cashflow.isChecked())
        
        self.config_manager.save_config()
    
    def browse_directory(self):
        """浏览目录"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择包含Excel文件的目录", self.dir_input.text()
        )
        if directory:
            self.dir_input.setText(directory)
    
    def edit_account_mapping(self):
        """编辑科目映射"""
        # TODO: 打开科目映射编辑对话框
        QMessageBox.information(self, "提示", "科目映射编辑功能开发中...")
    
    def reset_to_default(self):
        """重置为默认配置"""
        reply = QMessageBox.question(
            self, '确认', '确定要重置所有配置为默认值吗？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.config = self.config_manager.get_default_config()
            self.config_manager.save_config()
            self.load_config_to_ui()
            self.add_log("✅ 已重置为默认配置")
    
    def start_processing(self):
        """开始处理"""
        directory = self.dir_input.text()
        
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "错误", "请选择有效的目录！")
            return
        
        # 保存当前配置
        self.save_ui_to_config()
        
        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.edit_mapping_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 清空日志
        self.log_text.clear()
        self.add_log(f"📁 工作目录: {directory}")
        
        # 创建并启动处理线程
        self.process_thread = ProcessThread(directory, self.config_manager.config)
        self.process_thread.progress.connect(self.update_progress)
        self.process_thread.log.connect(self.add_log)
        self.process_thread.finished.connect(self.processing_finished)
        self.process_thread.start()
    
    def update_progress(self, current, total, message):
        """更新进度"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    
    def add_log(self, message):
        """添加日志"""
        self.log_text.append(message)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def processing_finished(self, success, message):
        """处理完成"""
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.edit_mapping_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        
        if success:
            self.status_label.setText("✅ 处理完成")
            QMessageBox.information(self, "成功", message)
        else:
            self.status_label.setText("❌ 处理失败")
            QMessageBox.critical(self, "错误", f"处理失败: {message}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
