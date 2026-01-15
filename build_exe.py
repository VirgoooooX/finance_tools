import os
import subprocess
import sys

def build():
    # 确保安装了 pyinstaller
    try:
        import PyInstaller
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    print("开始打包程序...")

    # 打包命令参数
    params = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--exclude-module", "matplotlib",
        "--exclude-module", "PIL",
        "--onefile",
        "--console", # Web 程序建议保留 console 以便查看后台日志，或者用 --windowed 隐藏
        "--name", "财务数据分析工具",
        # 包含 web 静态资源目录
        "--add-data", f"web{os.pathsep}web",
        # 包含默认配置文件（如果存在）
        "--add-data", f"config{os.pathsep}config",
        # 包含 tools 插件目录
        "--add-data", f"tools{os.pathsep}tools",
        # 包含 Web favicon/资源图标
        "--add-data", f"app_icon.ico{os.pathsep}.",
        # 入口文件
        "financial_analyzer.py"
    ]

    # 设置图标
    icon_path = "app_icon.ico"
    if os.path.exists(icon_path):
        params.insert(3, f"--icon={icon_path}")
    
    try:
        subprocess.check_call(params)
        print("\n" + "="*30)
        print("打包完成！")
        print(f"可执行文件位于: {os.path.join(os.getcwd(), 'dist', '财务数据分析工具.exe')}")
        print("="*30)
    except subprocess.CalledProcessError as e:
        print(f"打包失败: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    # 检查当前目录下是否有必要的文件
    if not os.path.exists("financial_analyzer.py"):
        print("错误：未找到 financial_analyzer.py")
        sys.exit(1)
    
    if not os.path.exists("web"):
        print("错误：未找到 web 静态资源目录")
        sys.exit(1)
    
    if not os.path.exists("config"):
        print("错误：未找到 config 配置目录")
        sys.exit(1)

    build()
