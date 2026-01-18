import os
import subprocess
import sys

def _ensure_pyinstaller() -> None:
    # 确保安装了 pyinstaller
    try:
        import PyInstaller
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])



def build(
    entry: str,
    name: str,
    onefile: bool,
    console: bool,
    icon_path: str,
    dry_run: bool,
) -> int:
    _ensure_pyinstaller()

    print("开始打包程序...")

    params = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--exclude-module", "matplotlib",
        "--exclude-module", "PIL",
        "--name", name,
        # 入口文件
        entry,
    ]

    params.append("--onefile" if onefile else "--onedir")
    params.append("--console" if console else "--windowed")

    if icon_path and os.path.exists(icon_path):
        params.append(f"--icon={icon_path}")

    for mod in ["uvicorn", "fastapi", "starlette"]:
        params.extend(["--collect-submodules", mod])
    for mod in [
        "tools.builtin_tools",
        "tools.report_ingestor.core",
        "tools.validation_report.core",
        "tools.financial_metrics.core",
    ]:
        params.extend(["--hidden-import", mod])

    if os.path.exists("web"):
        params.extend(["--add-data", f"web{os.pathsep}web"])
    if os.path.exists("tools"):
        params.extend(["--add-data", f"tools{os.pathsep}tools"])
    if os.path.exists("config"):
        params.extend(["--add-data", f"config{os.pathsep}config"])
    if os.path.exists("app_icon.ico"):
        params.extend(["--add-data", f"app_icon.ico{os.pathsep}."])

    print("PyInstaller 参数：")
    for p in params:
        print(" ", p)

    if dry_run:
        return 0

    try:
        subprocess.check_call(params)
        print("\n" + "="*30)
        print("打包完成！")
        if onefile:
            print(f"可执行文件位于: {os.path.join(os.getcwd(), 'dist', name + '.exe')}")
        else:
            print(f"目录位于: {os.path.join(os.getcwd(), 'dist', name)}")
        print("="*30)
    except subprocess.CalledProcessError as e:
        print(f"打包失败: {e}")
        return 1
    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyInstaller 打包脚本")
    parser.add_argument("--entry", default="financial_analyzer_web.py", help="入口文件")
    parser.add_argument("--name", default="财务数据分析工具", help="EXE 名称")
    parser.add_argument("--onedir", action="store_true", help="使用目录模式（默认 onefile）")
    parser.add_argument("--windowed", action="store_true", help="隐藏控制台窗口（默认 console）")
    parser.add_argument("--icon", default="app_icon.ico", help="图标路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印参数，不执行打包")
    args = parser.parse_args()

    # 检查当前目录下是否有必要的文件
    if not os.path.exists(args.entry):
        print(f"错误：未找到入口文件：{args.entry}")
        sys.exit(1)
    
    if not os.path.exists("web"):
        print("错误：未找到 web 静态资源目录")
        sys.exit(1)

    raise SystemExit(
        build(
            entry=args.entry,
            name=args.name,
            onefile=not bool(args.onedir),
            console=not bool(args.windowed),
            icon_path=args.icon,
            dry_run=bool(args.dry_run),
        )
    )
