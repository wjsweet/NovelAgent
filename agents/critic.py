# =============================================================================
# agents/critic.py —— 审核 Agent（全局设定比对 + 质量检查）
# =============================================================================
#
# 【审核 Agent 的职责】
# 审核是"第三棒"，也是整个反思循环（Reflection Loop）的核心。
# 它的工作是：拿着"宪法"（全局设定）去逐条比对草稿，找出问题。
#
# 【为什么审核 Agent 是解决幻觉问题的关键？】
#
# 大模型的幻觉问题在小说创作中有两种表现：
#   1. 事实性幻觉：主角名字写错、年龄前后不一致
#   2. 逻辑性幻觉：第1章说主角不善言辞，第3章突然变成演讲高手
#
# 审核 Agent 的解决方案：
#   - 把全局设定作为"检查清单"（Checklist）
#   - 逐条对照草稿，找出违反设定的地方
#   - 给出具体的修改意见（不是模糊的"写得不好"）
#
# 【反思循环（Reflection Loop）模式】
# 这是 Agent 开发中最重要的设计模式之一：
#
#   写手生成草稿 → 审核检查 → 发现问题 → 打回重写 → 再次审核 → ...
#
# 这个循环会一直进行，直到：
#   ① 审核通过（输出 PASS）
#   ② 达到最大修改轮次（防止死循环）
#
# 【类比自动化测试】
# 审核 Agent 就像你的"断言（Assert）"：
#   - 测试用例执行完后，你会断言"返回状态码是200"、"响应体包含 user_id"
#   - 审核 Agent 执行完后，会断言"主角名字是陈默"、"没有出现穿越情节"
# 如果断言失败，测试框架会报错并重试；审核失败，写手会重写。
#
# 【审核维度】
# 1. 人设一致性：主角姓名、性格、背景是否符合设定
# 2. 世界观一致性：时代背景、技术描写是否真实
# 3. 禁止元素检查：是否出现了禁止的情节元素
# 4. 逻辑连贯性：与历史章节是否有矛盾
# 5. 质量评估：文笔、情节、节奏是否达标
# =============================================================================

import os
import sys
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_critic_agent(llm: ChatOpenAI, config):
    """
    工厂函数：创建审核 Agent 节点函数。

    Args:
        llm: 大语言模型实例
        config: 全局配置实例

    Returns:
        符合 LangGraph 节点签名的函数
    """

    def critic_agent(state: dict) -> dict:
        """
        审核 Agent 节点函数。

        【执行流程】
        Step 1: 读取草稿和全局设定
        Step 2: 构建"检查清单式"Prompt
        Step 3: 调用 LLM 进行多维度审核
        Step 4: 解析审核结果（PASS / 修改意见）
        Step 5: 更新修改轮次计数器
        Step 6: 返回审核结果

        Args:
            state: LangGraph 共享状态字典

        Returns:
            dict: 包含审核反馈的状态更新
        """
        print("\n" + "=" * 60)
        print("🔍 [审核Agent] 开始工作")
        print("=" * 60)
        print("📋 职责：对照全局设定，多维度审核草稿质量")
        print("🎯 核心任务：防止人设崩塌、逻辑矛盾、幻觉内容")

        # ------------------------------------------------------------------
        # Step 1: 读取当前状态
        # ------------------------------------------------------------------
        print("\n📥 [Step 1] 读取审核材料...")

        novel_draft = state.get("novel_draft", "")
        outline = state.get("outline", "")
        current_chapter = state.get("current_chapter", 1)
        revision_round = state.get("revision_round", 0)
        max_rounds = config.novel.max_review_rounds
        key_events = state.get("key_events", [])
        chapter_summaries = state.get("chapter_summaries", [])

        print(f"   📖 审核章节：第 {current_chapter} 章")
        print(f"   🔄 当前修改轮次：{revision_round}/{max_rounds}")
        print(f"   📝 草稿长度：{len(novel_draft)} 字符")
        print(f"   🧠 历史关键事件：{len(key_events)} 个")

        # 安全检查：如果没有草稿，直接返回错误
        if not novel_draft:
            print("   ❌ 错误：没有草稿可以审核！")
            return {
                "review_feedback": "ERROR: 没有草稿可以审核",
                "review_passed": False,
                "current_step": "review_error"
            }

        # ------------------------------------------------------------------
        # Step 2: 检查是否达到最大修改轮次
        # ------------------------------------------------------------------
        print(f"\n🔢 [Step 2] 检查修改轮次限制...")

        if revision_round >= max_rounds:
            print(f"   ⚠️  已达到最大修改轮次（{max_rounds}轮），强制通过")
            print("   💡 原因：防止无限循环，保证系统能够终止")
            print("   💡 生产建议：可以增加人工审核环节，而不是强制通过")
            return {
                "review_feedback": f"PASS（已达最大修改轮次 {max_rounds}，强制通过）",
                "review_passed": True,
                "current_step": "review_passed_forced",
                "messages": [AIMessage(
                    content=f"[审核] 第{current_chapter}章强制通过（已达最大修改轮次）",
                    name="Critic"
                )]
            }

        # ------------------------------------------------------------------
        # Step 3: 构建"检查清单式"Prompt
        # ------------------------------------------------------------------
        print("\n📝 [Step 3] 构建审核 Prompt...")
        print("   💡 审核 Prompt 的设计原则：")
        print("      ① 把全局设定转成'检查清单'，逐条核对")
        print("      ② 要求给出具体问题，而不是模糊评价")
        print("      ③ 明确定义'通过'的标准（PASS 关键词）")

        global_setting = config.get_global_setting_prompt()
        c = config.character
        w = config.world

        # 构建历史一致性检查材料
        history_context = ""
        if key_events:
            history_context = "\n【已发生的关键事件（草稿不得与之矛盾）】\n"
            for event in key_events:
                history_context += f"  • {event}\n"

        # 构建禁止元素列表
        forbidden_list = "\n".join([f"  - {item}" for item in w.forbidden_elements])

        system_prompt = f"""你是一位严苛的小说主编，负责审核稿件质量。

{global_setting}

【你的审核职责】
你必须像一个"质量检查员"一样，逐条核对以下检查清单：

✅ 检查清单 1：人设一致性（最重要！）
  - 主角姓名是否始终是"{c.name}"？（不得出现其他名字）
  - 主角性格是否符合：{c.personality[:50]}...？
  - 主角职业背景是否符合：{c.occupation_start} → {c.occupation_end}？

✅ 检查清单 2：世界观一致性
  - 时代背景是否符合：{w.era}？
  - 技术描写是否真实可信？
  - 故事基调是否符合：{w.tone}？

✅ 检查清单 3：禁止元素检查
  以下元素绝对不能出现：
{forbidden_list}

✅ 检查清单 4：逻辑连贯性
  - 与大纲是否一致？
  - 与历史章节是否有矛盾？

✅ 检查清单 5：写作质量
  - 字数是否达到约 {config.novel.words_per_chapter} 字？
  - 情节是否推进了故事发展？
  - 是否有明显的逻辑漏洞？

【审核输出格式】
如果所有检查项都通过，请回复：
PASS
（简短说明通过原因）

如果有问题，请回复：
REVISE
问题1：[具体描述问题，指出在哪里出现的]
问题2：[...]
修改建议：[具体告诉写手如何修改]

【重要规则】
- 不要给模糊的评价（如"写得不够好"），必须指出具体问题
- 如果只有小问题（如个别用词），可以直接通过（PASS）
- 如果有人设崩塌或禁止元素，必须打回重写（REVISE）
"""

        human_message = f"""请审核以下草稿：

【故事大纲（参考）】
{outline[:500]}...

{history_context}

【待审核草稿（第{current_chapter}章）】
{novel_draft}

请按照检查清单逐条审核，给出明确的 PASS 或 REVISE 结论。"""

        print(f"   ✅ 审核 Prompt 构建完成")
        print(f"   📏 System Prompt：{len(system_prompt)} 字符")
        print(f"   📏 Human Message：{len(human_message)} 字符")

        # ------------------------------------------------------------------
        # Step 4: 调用 LLM 进行审核
        # ------------------------------------------------------------------
        print(f"\n🤖 [Step 4] 调用大模型进行审核...")
        print(f"   🔧 模型：{config.llm.model_name}")
        print("   ⏳ 等待审核结果...")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]

        try:
            response = llm.invoke(messages)
            feedback = response.content
            print(f"   ✅ 审核完成！反馈长度：{len(feedback)} 字符")
        except Exception as e:
            print(f"   ❌ LLM 调用失败：{e}")
            feedback = f"审核失败：{str(e)}"

        # ------------------------------------------------------------------
        # Step 5: 解析审核结果
        # ------------------------------------------------------------------
        print("\n🔎 [Step 5] 解析审核结果...")

        feedback_upper = feedback.upper()
        is_passed = "PASS" in feedback_upper and "REVISE" not in feedback_upper

        print(f"   📋 审核结论：{'✅ 通过 (PASS)' if is_passed else '❌ 需要修改 (REVISE)'}")
        print(f"\n   审核详情：")
        print("   " + "-" * 40)
        # 打印审核反馈（每行加缩进）
        for line in feedback.split("\n")[:15]:  # 最多打印15行
            if line.strip():
                print(f"   {line}")
        if len(feedback.split("\n")) > 15:
            print(f"   ...（共 {len(feedback.split(chr(10)))} 行）")
        print("   " + "-" * 40)

        # ------------------------------------------------------------------
        # Step 6: 更新修改轮次
        # ------------------------------------------------------------------
        print(f"\n📊 [Step 6] 更新状态...")

        new_revision_round = revision_round + 1 if not is_passed else revision_round

        if is_passed:
            print(f"   ✅ 审核通过！第 {current_chapter} 章定稿")
            print(f"   📈 本章共经历 {revision_round} 轮修改")
            current_step = "review_passed"
        else:
            print(f"   🔄 审核未通过，进入第 {new_revision_round} 轮修改")
            print(f"   ⚠️  剩余修改机会：{max_rounds - new_revision_round} 次")
            current_step = "review_failed"

        print("\n✅ [审核Agent] 工作完成！")
        if is_passed:
            print("   ➡️  下一步：流程结束（或进入下一章）")
        else:
            print("   ➡️  下一步：写手 Agent 将根据审核意见修改正文")

        # ------------------------------------------------------------------
        # 返回更新的状态
        # ------------------------------------------------------------------
        return {
            "review_feedback": feedback,
            "review_passed": is_passed,
            "revision_round": new_revision_round,
            "current_step": current_step,
            "messages": [AIMessage(
                content=f"[审核] {'PASS' if is_passed else 'REVISE'} - 第{current_chapter}章审核完成",
                name="Critic"
            )]
        }

    return critic_agent


def create_review_router(config):
    """
    工厂函数：创建路由函数。

    【路由函数的作用】
    在 LangGraph 中，路由函数决定"下一步去哪个节点"。
    这是实现"反思循环"的关键机制。

    【为什么要把路由逻辑单独提取出来？】
    1. 职责分离：审核 Agent 负责"判断"，路由函数负责"决策"
    2. 可测试性：路由逻辑可以单独测试，不依赖 LLM
    3. 可扩展性：未来可以添加更复杂的路由逻辑（如多个审核员投票）

    Args:
        config: 全局配置实例

    Returns:
        路由函数
    """

    def review_router(state: dict) -> Literal["writer", "end"]:
        """
        根据审核结果决定下一步。

        【决策逻辑】
        - 审核通过（PASS）→ 结束流程
        - 审核未通过（REVISE）→ 打回给写手重写
        - 达到最大轮次 → 强制结束（防止死循环）

        Args:
            state: 当前全局状态

        Returns:
            "writer": 打回重写
            "end": 结束流程
        """
        print("\n" + "=" * 60)
        print("🔀 [路由决策] 根据审核结果决定下一步")
        print("=" * 60)

        review_passed = state.get("review_passed", False)
        revision_round = state.get("revision_round", 0)
        max_rounds = config.novel.max_review_rounds
        current_chapter = state.get("current_chapter", 1)

        print(f"   📊 审核结果：{'通过' if review_passed else '未通过'}")
        print(f"   🔢 修改轮次：{revision_round}/{max_rounds}")

        if review_passed:
            print(f"   ✅ 决策：结束流程（第 {current_chapter} 章已通过审核）")
            print("   💡 LangGraph 将执行 END 节点，流程终止")
            return "end"
        elif revision_round >= max_rounds:
            print(f"   ⚠️  决策：强制结束（已达最大修改轮次 {max_rounds}）")
            print("   💡 这是防止死循环的安全机制")
            return "end"
        else:
            print(f"   🔄 决策：打回重写（第 {revision_round} 轮修改）")
            print("   💡 LangGraph 将回到 Writer 节点，写手重新撰写")
            return "writer"

    return review_router


# =============================================================================
# 调试入口：单独测试审核 Agent
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 审核 Agent 独立测试")
    print("=" * 60)

    from config import app_config

    llm = ChatOpenAI(
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        model=app_config.llm.model_name,
        temperature=app_config.llm.temperature
    )

    critic = create_critic_agent(llm, app_config)
    router = create_review_router(app_config)

    # 测试1：正常草稿（应该通过）
    test_state_pass = {
        "messages": [],
        "novel_draft": """
第一章：迷茫的起点

陈默盯着屏幕上密密麻麻的 pytest 报告，手指无意识地敲击着桌面。
28岁，4年工作经验，精通 Selenium 和 pytest，在公司的自动化测试团队里算是骨干。
但今天，他感到一种说不清道不明的倦怠。

"又是一堆回归测试。"他低声自语，眼神空洞地扫过那些绿色的 PASSED 标记。

转机出现在下午三点。他在 V2EX 上刷到了一篇帖子：《我用 LangChain 做了一个自动分析竞品的 Agent》。
""",
        "outline": "陈默是测试工程师，决定转型Agent开发",
        "review_feedback": "",
        "current_chapter": 1,
        "revision_round": 0,
        "key_events": [],
        "chapter_summaries": []
    }

    print("\n--- 测试1：正常草稿 ---")
    result = critic(test_state_pass)
    print(f"\n路由决策：{router(result)}")

    # 测试2：人设崩塌的草稿（应该打回）
    test_state_fail = {
        **test_state_pass,
        "novel_draft": """
第一章：迷茫的起点

李明是一个天才程序员，从小就展现出超凡的编程天赋。
他穿越到了2024年，凭借未来的知识，轻松掌握了所有AI技术。
""",
        "revision_round": 0
    }

    print("\n--- 测试2：人设崩塌草稿（应该打回）---")
    result2 = critic(test_state_fail)
    print(f"\n路由决策：{router(result2)}")
