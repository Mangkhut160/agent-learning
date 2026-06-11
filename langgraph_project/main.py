"""
LangGraph 学习项目主入口
包含两个 Demo：
1. 基础状态机工作流（规划者-执行者循环）
2. 三步问答助手（理解→搜索→回答）
"""

import sys
import os

# 设置 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_demo1():
    """运行 Demo 1: 基础状态机工作流"""
    from src.demo1_basic_workflow import run_demo
    run_demo("帮我设计一个简单的待办事项应用")


def run_demo2(query: str = None):
    """运行 Demo 2: 三步问答助手"""
    from src.demo2_search_assistant import run_demo

    if query:
        run_demo(query)
    else:
        run_demo("最近人工智能领域有什么重大突破？")


def run_demo2_interactive():
    """运行 Demo 2 交互模式"""
    print("\n交互模式：直接运行 demo2 并输入问题")
    query = input("请输入问题: ").strip()
    if query:
        from src.demo2_search_assistant import run_demo
        run_demo(query)


def main():
    print("\n" + "="*60)
    print("LangGraph 学习项目")
    print("="*60)
    print("\n可用选项:")
    print("  1. Demo 1: 基础状态机工作流")
    print("  2. Demo 2: 三步问答助手")
    print("  3. 退出")
    print("-"*60)

    choice = input("\n请选择 [1-3]: ").strip()

    if choice == "1":
        run_demo1()
    elif choice == "2":
        query = input("请输入问题（回车使用默认）: ").strip()
        run_demo2(query if query else None)
    elif choice == "3":
        print("再见！")
    else:
        print("无效选择")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "demo1":
            run_demo1()
        elif arg == "demo2":
            query = sys.argv[2] if len(sys.argv) > 2 else None
            run_demo2(query)
        elif arg == "interactive":
            run_demo2_interactive()
        else:
            print(f"未知参数: {arg}")
            print("用法: python main.py [demo1|demo2|interactive]")
    else:
        main()
