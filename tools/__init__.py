# =============================================================================
# tools/__init__.py —— 工具模块（LangChain Tool 规范）
# =============================================================================
#
# 【什么是 LangChain Tool？】
# Tool 是 Agent 可以"调用"的外部能力，是 Agent 与外部世界交互的接口。
# 常见的 Tool 包括：
#   - 搜索引擎（Google Search）
#   - 代码执行器（Python REPL）
#   - 数据库查询
#   - 文件读写
#   - 自定义业务逻辑
#
# 【Tool 的标准格式（LangChain 规范）】
# 使用 @tool 装饰器定义工具，LangChain 会自动：
#   1. 解析函数的 docstring 作为工具描述（告诉 LLM 这个工具是干什么的）
#   2. 解析函数参数作为工具的输入 schema
#   3. 把函数返回值作为工具的输出
#
# 【为什么当前项目的 tools 目录是空的？】
# 在当前架构中，我们使用的是"固定流水线"模式（Pipeline），
# 而不是"工具调用"模式（Tool Calling）。
# 区别：
#   - Pipeline 模式：流程固定，Designer → Writer → Critic，顺序执行
#   - Tool Calling 模式：Agent 自主决定调用哪些工具，更灵活但更复杂
#
# 当前阶段用 Pipeline 模式更适合学习，因为：
#   1. 流程清晰，容易理解和调试
#   2. 输出可预期，便于测试
#   3. 适合有明确步骤的任务（小说创作流程是固定的）
#
# 【未来扩展方向】
# 可以在这里添加：
#   - 网络搜索工具（搜索最新的 AI 技术资讯）
#   - 字数统计工具
#   - 风格检查工具
#   - 自动保存工具
#
# 【类比自动化测试】
# Tool 就像你测试框架中的"工具函数"（utils），
# 比如 take_screenshot()、get_element_text()、wait_for_element()，
# 这些函数封装了底层操作，供测试用例调用。
# =============================================================================

from langchain_core.tools import tool
from typing import Optional
import os


@tool
def word_count_tool(text: str) -> str:
    """
    统计文本的字数和字符数。
    用于检查小说章节是否达到目标字数。

    Args:
        text: 需要统计的文本内容

    Returns:
        包含字数统计信息的字符串
    """
    if not text:
        return "文本为空"

    # 中文字数统计（去除空格和标点）
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)
    lines = len(text.split('\n'))

    return (
        f"📊 字数统计：\n"
        f"  - 中文字符：{chinese_chars} 字\n"
        f"  - 总字符数：{total_chars} 字符\n"
        f"  - 行数：{lines} 行"
    )


@tool
def save_chapter_tool(chapter_num: int, content: str, output_dir: str = "./output") -> str:
    """
    将章节内容保存到文件。
    用于持久化已审核通过的章节。

    Args:
        chapter_num: 章节编号
        content: 章节内容
        output_dir: 输出目录路径

    Returns:
        保存结果信息
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"chapter_{chapter_num:02d}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        file_size = os.path.getsize(file_path)
        return f"✅ 第{chapter_num}章已保存：{file_path}（{file_size} 字节）"
    except Exception as e:
        return f"❌ 保存失败：{str(e)}"


@tool
def check_character_consistency_tool(text: str, character_name: str) -> str:
    """
    检查文本中主角姓名的一致性。
    这是一个简单的规则检查工具，用于辅助审核 Agent。

    Args:
        text: 需要检查的文本
        character_name: 正确的主角姓名

    Returns:
        检查结果
    """
    if not text or not character_name:
        return "参数不完整"

    # 检查主角名字出现次数
    count = text.count(character_name)

    # 检查是否有其他可能的名字（简单启发式）
    # 实际项目中可以用 NER（命名实体识别）来做更精确的检查
    issues = []
    if count == 0:
        issues.append(f"⚠️  主角姓名 '{character_name}' 在文本中未出现，请检查是否使用了其他名字")

    if issues:
        return "\n".join(issues)
    else:
        return f"✅ 主角姓名检查通过：'{character_name}' 出现 {count} 次"


# 导出所有工具
AVAILABLE_TOOLS = [
    word_count_tool,
    save_chapter_tool,
    check_character_consistency_tool,
]

__all__ = [
    "word_count_tool",
    "save_chapter_tool",
    "check_character_consistency_tool",
    "AVAILABLE_TOOLS",
]
