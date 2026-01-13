"""
财务数据分析工具 - CustomTkinter GUI (现代化美观版本)
使用CustomTkinter实现更精美的界面效果
"""
import sys
import os
import threading
import customtkinter as ctk
from tkinter import filedialog
from config_manager import ConfigManager


class FinancialAnalyzerApp(ctk.CTk):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 配置管理器
        self.config_manager = ConfigManager()
        self.processing = False
        
        # 窗口配置
        self.title("财务数据分析工具 v2.0")
        self.geometry("900x750")
        
        # 设置主题
        ctk.set_appearance_mode("light")  # 可选: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # 可选: "blue", "green", "dark-blue"
        
        # 创建UI
        self.create_ui()
        
    def create_ui(self):
        """创建用户界面"""
        # 主容器
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ========== 标题区域 ==========
        title_frame = ctk.CTkFrame(main_container, corner_radius=15, fg_color=("#667eea", "#764ba2"))
        title_frame.pack(fill="x", pady=(0, 20))
        
        title = ctk.CTkLabel(
            title_frame, 
            text="📊 财务数据分析工具",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="white"
        )
        title.pack(pady=(20, 5))
        
        subtitle = ctk.CTkLabel(
            title_frame,
            text="智能财务报表处理与分析系统",
            font=ctk.CTkFont(size=14),
            text_color="white"
        )
        subtitle.pack(pady=(0, 20))
        
        # ========== 文件选择区域 ==========
        file_frame = ctk.CTkFrame(main_container, corner_radius=12)
        file_frame.pack(fill="x", pady=(0, 15))
        
        file_label = ctk.CTkLabel(
            file_frame,
            text="📁 数据源",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        file_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        file_input_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        file_input_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.dir_entry = ctk.CTkEntry(
            file_input_frame,
            placeholder_text="选择包含Excel文件的目录...",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        self.dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.dir_entry.insert(0, os.getcwd())
        
        browse_btn = ctk.CTkButton(
            file_input_frame,
            text="📂 浏览",
            command=self.browse_directory,
            width=100,
            height=40,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        browse_btn.pack(side="right")
        
        # ========== 配置选项区域 ==========
        config_frame = ctk.CTkFrame(main_container, corner_radius=12)
        config_frame.pack(fill="x", pady=(0, 15))
        
        config_label = ctk.CTkLabel(
            config_frame,
            text="⚙️ 配置选项",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        config_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # 输出选项
        output_frame = ctk.CTkFrame(config_frame, fg_color=("#f0f0f0", "#2b2b2b"))
        output_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        output_title = ctk.CTkLabel(output_frame, text="输出内容:", font=ctk.CTkFont(weight="bold"))
        output_title.pack(side="left", padx=10, pady=10)
        
        self.cb_original = ctk.CTkCheckBox(output_frame, text="原始数据")
        self.cb_original.pack(side="left", padx=5)
        self.cb_original.select()
        
        self.cb_validation = ctk.CTkCheckBox(output_frame, text="验证报告")
        self.cb_validation.pack(side="left", padx=5)
        self.cb_validation.select()
        
        self.cb_metrics = ctk.CTkCheckBox(output_frame, text="财务指标")
        self.cb_metrics.pack(side="left", padx=5)
        self.cb_metrics.select()
        
        # 验证选项
        validation_frame = ctk.CTkFrame(config_frame, fg_color=("#f0f0f0", "#2b2b2b"))
        validation_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.cb_enable_validation = ctk.CTkCheckBox(
            validation_frame, 
            text="启用会计恒等式验证"
        )
        self.cb_enable_validation.pack(side="left", padx=10, pady=10)
        self.cb_enable_validation.select()
        
        tolerance_label = ctk.CTkLabel(validation_frame, text="容差:")
        tolerance_label.pack(side="left", padx=(20, 5))
        
        self.tolerance_entry = ctk.CTkEntry(validation_frame, width=80)
        self.tolerance_entry.pack(side="left")
        self.tolerance_entry.insert(0, "0.01")
        
        # 指标选项
        metrics_frame = ctk.CTkFrame(config_frame, fg_color=("#f0f0f0", "#2b2b2b"))
        metrics_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        metrics_title = ctk.CTkLabel(metrics_frame, text="计算指标:", font=ctk.CTkFont(weight="bold"))
        metrics_title.pack(side="left", padx=10, pady=10)
        
        self.cb_liquidity = ctk.CTkCheckBox(metrics_frame, text="流动性")
        self.cb_liquidity.pack(side="left", padx=5)
        self.cb_liquidity.select()
        
        self.cb_solvency = ctk.CTkCheckBox(metrics_frame, text="偿债能力")
        self.cb_solvency.pack(side="left", padx=5)
        self.cb_solvency.select()
        
        self.cb_profitability = ctk.CTkCheckBox(metrics_frame, text="盈利能力")
        self.cb_profitability.pack(side="left", padx=5)
        self.cb_profitability.select()
        
        self.cb_cashflow = ctk.CTkCheckBox(metrics_frame, text="现金流")
        self.cb_cashflow.pack(side="left", padx=5)
        self.cb_cashflow.select()
        
        # ========== 操作按钮 ==========
        button_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        button_frame.pack(fill="x", pady=(0, 15))
        
        edit_btn = ctk.CTkButton(
            button_frame,
            text="📝 编辑科目映射",
            command=self.edit_mapping,
            fg_color="#f39c12",
            hover_color="#e67e22",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        edit_btn.pack(side="left", padx=(0, 10))
        
        reset_btn = ctk.CTkButton(
            button_frame,
            text="🔄 重置配置",
            command=self.reset_config,
            fg_color="#95a5a6",
            hover_color="#7f8c8d",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        reset_btn.pack(side="left")
        
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="▶️ 开始处理",
            command=self.start_processing,
            fg_color="#2ecc71",
            hover_color="#27ae60",
            height=50,
            width=200,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.start_btn.pack(side="right")
        
        # ========== 进度条 ==========
        self.progress = ctk.CTkProgressBar(main_container, height=20)
        self.progress.pack(fill="x", pady=(0, 10))
        self.progress.set(0)
        
        # ========== 状态标签 ==========
        self.status_label = ctk.CTkLabel(
            main_container,
            text="就绪",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#27ae60"
        )
        self.status_label.pack(pady=(0, 15))
        
        # ========== 日志区域 ==========
        log_frame = ctk.CTkFrame(main_container, corner_radius=12)
        log_frame.pack(fill="both", expand=True)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="📋 处理日志",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        log_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        self.log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(size=11),
            wrap="word"
        )
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    def browse_directory(self):
        """浏览目录"""
        directory = filedialog.askdirectory(
            title="选择包含Excel文件的目录",
            initialdir=self.dir_entry.get()
        )
        if directory:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, directory)
    
    def edit_mapping(self):
        """编辑科目映射"""
        self.log_text.insert("end", "📝 科目映射编辑功能开发中...\n")
    
    def reset_config(self):
        """重置配置"""
        self.config_manager.config = self.config_manager.get_default_config()
        self.config_manager.save_config()
        self.log_text.insert("end", "✅ 已重置为默认配置\n")
    
    def start_processing(self):
        """开始处理"""
        if self.processing:
            return
        
        directory = self.dir_entry.get()
        if not os.path.isdir(directory):
            self.log_text.insert("end", "❌ 请选择有效的目录！\n")
            return
        
        self.processing = True
        self.start_btn.configure(state="disabled", text="处理中...")
        self.progress.set(0)
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", f"📁 工作目录: {directory}\n")
        
        # 在后台线程运行
        thread = threading.Thread(target=self.process_data, args=(directory,))
        thread.daemon = True
        thread.start()
    
    def process_data(self, directory):
        """处理数据"""
        try:
            import financial_analyzer
            
            def progress_callback(current, total, message):
                self.after(0, self.update_progress, current, total, message)
            
            analyzer = financial_analyzer.FinancialAnalyzer(
                config=self.config_manager.config,
                progress_callback=progress_callback
            )
            
            analyzer.process_directory(directory)
            
            self.after(0, self.processing_complete, True)
        except Exception as e:
            self.after(0, self.processing_complete, False, str(e))
    
    def update_progress(self, current, total, message):
        """更新进度"""
        if total > 0:
            self.progress.set(current / total)
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.status_label.configure(text=message)
    
    def processing_complete(self, success, error=None):
        """处理完成"""
        self.processing = False
        self.start_btn.configure(state="normal", text="▶️ 开始处理")
        
        if success:
            self.status_label.configure(text="✅ 处理完成", text_color="#27ae60")
            self.log_text.insert("end", "\n✅ 所有处理完成！\n")
        else:
            self.status_label.configure(text="❌ 处理失败", text_color="#e74c3c")
            self.log_text.insert("end", f"\n❌ 错误: {error}\n")


def main():
    """主函数"""
    app = FinancialAnalyzerApp()
    app.mainloop()


if __name__ == '__main__':
    main()
