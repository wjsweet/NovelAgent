# =============================================================================
# agents/writer.py —— 写手 Agent（正文撰写 + 记忆注入）
# =============================================================================
#
# 【写手 Agent 的职责】
# 写手是"第二棒"，接收设计师的大纲，撰写具体的小说正文。
# 核心挑战：如何让写手在写第5章时，还记得第1章发生了什么？
#
# 【"记忆"在 Agent 中的实现方式】
#
# 大模型本身是"无状态"的——每次调用都是全新的对话，它不记得上次说了什么。
# 这就是为什么长篇小说会"人设崩塌"的根本原因。
#
# 解决方案：把"记忆"外化到状态（State）中，每次调用时把历史内容注入 Prompt。
#
# 具体实现：
#   - 短期记忆：把最近几章的内容摘要放入 Prompt（避免 Token 超限）
#   - 长期记忆：全局设定（主角档案、世界观）永远注入，不会遗忘
#   - 情节记忆：已发生的关键事件列表，防止前后矛盾
#
# 【类比自动化测试】
# 就像你在做 E2E 测试时，每个测试步骤都需要知道"前面步骤的结果"，
# 写手的"记忆注入"就是把前面章节的"测试结果"传递给当前步骤。
#
# 【记忆管理的三个层次】
# Level 1 - 全局设定（永久记忆）：主角名字、性格、世界观 → 每次都注入
# Level 2 - 章节摘要（中期记忆）：已写章节的摘要 → 注入最近3章
# Level 3 - 当前大纲（工作记忆）：当前章节的大纲 → 每次都注入
# =============================================================================

import os
import sys
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract_chapter_summary(chapter_content: str, max_length: int = 300) -> str:
    """
    从章节内容中提取摘要。

    【为什么需要摘要而不是全文？】
    大模型的上下文窗口有限（如 glm-4-flash 约 128K tokens）。
    如果把所有已写章节的全文都塞进 Prompt，很快就会超限。
    解决方案：只保留每章的"关键信息摘要"，而不是全文。

    这是 Agent 开发中"记忆压缩"的常见技术。

    Args:
        chapter_content: 章节全文
        max_length: 摘要最大长度

    Returns:
        str: 章节摘要
    """
    if not chapter_content:
        return ""

    # 简单摘要：取前 max_length 字符
    # 生产环境中可以用 LLM 生成更智能的摘要
    if len(chapter_content) <= max_length:
        return chapter_content

    # 尝试在句子边界截断
    truncated = chapter_content[:max_length]
    last_period = max(
        truncated.rfind("。"),
        truncated.rfind("！"),
        truncated.rfind("？")
    )
    if last_period > max_length * 0.7:  # 如果句子边界在合理位置
        return truncated[:last_period + 1] + "（摘要）"

    return truncated + "...（摘要）"


def _build_memory_context(state: dict, config) -> str:
    """
    构建"记忆上下文"，注入写手的 Prompt。

    【这是解决幻觉问题的核心函数】

    记忆上下文包含：
    1. 已写章节的摘要（防止前后矛盾）
    2. 关键情节节点（已发生的重要事件）
    3. 主角当前状态（情绪、位置、关系）

    Args:
        state: 当前全局状态
        config: 全局配置

    Returns:
        str: 格式化的记忆上下文
    """
    chapter_summaries = state.get("chapter_summaries", [])
    key_events = state.get("key_events", [])
    current_chapter = state.get("current_chapter", 1)

    if not chapter_summaries and not key_events:
        return "（这是第一章，暂无历史记忆）"

    memory_text = f"【已发生的故事（第1章到第{current_chapter - 1}章摘要）】\n\n"

    # 只保留最近3章的摘要（避免 Prompt 过长）
    recent_summaries = chapter_summaries[-3:] if len(chapter_summaries) > 3 else chapter_summaries
    for i, summary in enumerate(recent_summaries):
        chapter_num = len(chapter_summaries) - len(recent_summaries) + i + 1
        memory_text += f"第{chapter_num}章摘要：{summary}\n\n"

    if key_events:
        memory_text += "【关键情节节点（不可违背）】\n"
        for event in key_events[-5:]:  # 最近5个关键事件
            memory_text += f"  • {event}\n"

    return memory_text.strip()


def create_writer_agent(llm: ChatOpenAI, config):
    """
    工厂函数：创建写手 Agent 节点函数。

    Args:
        llm: 大语言模型实例
        config: 全局配置实例

    Returns:
        符合 LangGraph 节点签名的函数
    """

    def writer_agent(state: dict) -> dict:
        """
        写手 Agent 节点函数。

        【执行流程】
        Step 1: 读取当前状态（大纲、章节数、历史记忆）
        Step 2: 构建记忆上下文（防止人设崩塌的关键）
        Step 3: 构建包含全局设定 + 记忆 + 大纲的 Prompt
        Step 4: 调用 LLM 撰写正文
        Step 5: 提取关键事件，更新记忆
        Step 6: 返回更新的状态

        Args:
            state: LangGraph 共享状态字典

        Returns:
            dict: 需要更新的状态字段
        """
        print("\n" + "=" * 60)
        print("✍️  [写手Agent] 开始工作")
        print("=" * 60)
        print("📋 职责：根据大纲 + 历史记忆，撰写当前章节正文")

        # ------------------------------------------------------------------
        # Step 1: 读取当前状态
        # ------------------------------------------------------------------
        print("\n📥 [Step 1] 读取当前状态...")

        outline = state.get("outline", "")
        current_chapter = state.get("current_chapter", 1)
        review_feedback = state.get("review_feedback", "")
        revision_round = state.get("revision_round", 0)
        existing_draft = state.get("novel_draft", "")

        print(f"   📖 当前章节：第 {current_chapter} 章")
        print(f"   🔄 修改轮次：第 {revision_round} 轮")
        print(f"   📝 大纲长度：{len(outline)} 字符")

        if review_feedback and revision_round > 0:
            print(f"   📋 审核反馈：{review_feedback[:80]}...")
            print("   💡 写手将根据审核意见修改正文")

        # ------------------------------------------------------------------
        # Step 2: 构建记忆上下文
        # ------------------------------------------------------------------
        print("\n🧠 [Step 2] 构建记忆上下文（注入历史记忆）...")
        print("   💡 原理：把已写章节的摘要注入 Prompt，让模型'记住'前面发生了什么")
        print("   💡 这是解决'人设崩塌'和'情节矛盾'的核心技术")

        memory_context = _build_memory_context(state, config)

        chapter_summaries = state.get("chapter_summaries", [])
        key_events = state.get("key_events", [])
        print(f"   📚 历史章节摘要数：{len(chapter_summaries)}")
        print(f"   🎯 关键事件数：{len(key_events)}")
        print(f"   📏 记忆上下文长度：{len(memory_context)} 字符")

        # ------------------------------------------------------------------
        # Step 3: 构建 Prompt
        # ------------------------------------------------------------------
        print("\n📝 [Step 3] 构建写作 Prompt...")
        print("   💡 Prompt 结构：全局设定（宪法）+ 历史记忆 + 大纲 + 写作指令")

        global_setting = config.get_global_setting_prompt()
        target_words = config.novel.words_per_chapter

        system_prompt = f"""你是一位文笔极佳的小说家，专注于职场成长类小说的创作。

{global_setting}

【写作原则】
1. 严格遵守全局设定，主角 {config.character.name} 的姓名、性格、背景不得更改
2. 情节必须与历史记忆保持一致，不得出现前后矛盾
3. 每章约 {target_words} 字，情感细腻，细节丰富
4. 技术描写要真实可信，避免外行描述
5. 对话要符合人物性格，{config.character.name} 说话内敛、精准，不废话

【写作风格】
- 第三人称叙事
- 现实主义笔触，避免夸张
- 技术场景要有代入感（如：写代码的专注状态、调试时的挫败感）
- 心理描写要细腻，展现主角的内心成长

【历史记忆（必须保持一致！）】
{memory_context}
"""

        # 根据是否有审核反馈，构建不同的任务指令
        if review_feedback and revision_round > 0:
            print(f"   🔄 模式：修改模式（第 {revision_round} 轮）")
            human_message = f"""请根据审核意见修改第 {current_chapter} 章：

【故事大纲】
{outline}

【上一版草稿】
{existing_draft[:1000] if existing_draft else '（无）'}...

【审核意见（必须解决这些问题！）】
{review_feedback}

请针对审核意见中的每个问题进行修改，保持其他优秀部分不变。
修改后的章节应该约 {target_words} 字。"""
        else:
            print(f"   ✍️  模式：首次撰写模式")
            human_message = f"""请根据以下大纲撰写第 {current_chapter} 章：

【故事大纲】
{outline}

【写作要求】
- 本章约 {target_words} 字
- 重点展现主角 {config.character.name} 在这一阶段的心理状态和成长
- 技术细节要真实（如果涉及编程，代码逻辑要正确）
- 结尾要有悬念或情感钩子，吸引读者继续阅读

请开始撰写第 {current_chapter} 章正文："""

        print(f"   ✅ Prompt 构建完成")
        print(f"   📏 System Prompt：{len(system_prompt)} 字符")
        print(f"   📏 Human Message：{len(human_message)} 字符")
        print(f"   📏 总 Prompt 长度：{len(system_prompt) + len(human_message)} 字符")

        # ------------------------------------------------------------------
        # Step 4: 调用 LLM 撰写正文
        # ------------------------------------------------------------------
        print(f"\n🤖 [Step 4] 调用大模型撰写第 {current_chapter} 章...")
        print(f"   🔧 模型：{config.llm.model_name} | 温度：{config.llm.temperature}")
        print("   ⏳ 等待模型响应（正文较长，可能需要30-60秒）...")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]

        try:
            response = llm.invoke(messages)
            novel_draft = response.content
            print(f"   ✅ 正文生成成功！字数：{len(novel_draft)} 字符")
        except Exception as e:
            print(f"   ❌ LLM 调用失败：{e}")
            novel_draft = f"正文生成失败：{str(e)}"

        # ------------------------------------------------------------------
        # Step 5: 提取关键事件，更新记忆
        # ------------------------------------------------------------------
        print("\n🧠 [Step 5] 更新记忆系统...")
        print("   💡 把本章的关键事件提取出来，存入'长期记忆'")
        print("   💡 下一章写作时，这些事件会被注入 Prompt，防止前后矛盾")

        # 生成本章摘要
        chapter_summary = _extract_chapter_summary(novel_draft)
        print(f"   📝 本章摘要（{len(chapter_summary)} 字符）：{chapter_summary[:80]}...")

        # 更新章节摘要列表
        updated_summaries = list(state.get("chapter_summaries", []))
        updated_summaries.append(chapter_summary)

        # 提取关键事件（简单版：从摘要中提取第一句话作为关键事件）
        # 生产环境中可以用 LLM 专门提取关键事件
        updated_events = list(state.get("key_events", []))
        if chapter_summary:
            first_sentence_end = max(
                chapter_summary.find("。"),
                chapter_summary.find("！"),
                chapter_summary.find("？")
            )
            if first_sentence_end > 0:
                key_event = f"第{current_chapter}章：{chapter_summary[:first_sentence_end + 1]}"
                updated_events.append(key_event)
                print(f"   🎯 新增关键事件：{key_event[:60]}...")

        # ------------------------------------------------------------------
        # Step 6: 打印结果预览
        # ------------------------------------------------------------------
        print("\n📄 [Step 6] 正文预览（前300字）：")
        print("-" * 40)
        print(novel_draft[:300] + "..." if len(novel_draft) > 300 else novel_draft)
        print("-" * 40)

        print("\n✅ [写手Agent] 工作完成！")
        print(f"   📊 输出：第 {current_chapter} 章正文（{len(novel_draft)} 字符）")
        print(f"   🧠 记忆更新：{len(updated_summaries)} 章摘要，{len(updated_events)} 个关键事件")
        print("   ➡️  下一步：审核 Agent 将检查正文质量")

        # ------------------------------------------------------------------
        # 返回更新的状态
        # ------------------------------------------------------------------
        return {
            "novel_draft": novel_draft,
            "chapter_summaries": updated_summaries,
            "key_events": updated_events,
            "current_step": "writing_complete",
            "messages": [AIMessage(
                content=f"[写手] 第{current_chapter}章已完成（{len(novel_draft)}字）",
                name="Writer"
            )]
        }

    return writer_agent


# =============================================================================
# 调试入口：单独测试写手 Agent
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 写手 Agent 独立测试")
    print("=" * 60)

    from config import app_config

    llm = ChatOpenAI(
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        model=app_config.llm.model_name,
        temperature=app_config.llm.temperature
    )

    writer = create_writer_agent(llm, app_config)

    # 模拟有历史记忆的状态（测试记忆注入功能）
    test_state = {
        "messages": [HumanMessage(content="开始写作")],
        "novel_draft": "",
        "outline": """
## 故事大纲
### 第一阶段：起因（第1-2章）
- 核心事件：陈默在公司做了4年测试，感到迷茫
- 关键转折：偶然看到 AI Agent 演示，被震撼
### 第二阶段：发展（第3-6章）
- 核心事件：陈默开始业余学习 LangChain
- 遇到的挫折：第一个 Agent 项目失败
""",
        "review_feedback": "",
        "current_step": "design_complete",
        "current_chapter": 2,
        "revision_round": 0,
        # 模拟第1章已写完的记忆
        "chapter_summaries": [
            "陈默是一名28岁的自动化测试工程师，在公司工作了4年，精通pytest和Selenium。某天他在技术论坛看到了一个AI Agent的演示视频，被深深震撼，决定转型。"
        ],
        "key_events": [
            "第1章：陈默在技术论坛看到AI Agent演示，决定转型学习Agent开发。"
        ]
    }

    result = writer(test_state)
    print("\n" + "=" * 60)
    print("📋 完整正文：")
    print("=" * 60)
    print(result["novel_draft"])
