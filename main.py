# =============================================================================
# main.py —— 小说创作多智能体系统 · 主入口
# =============================================================================
#
# 【系统架构总览】
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │                    NovelAgent 系统架构                        │
#   ├─────────────────────────────────────────────────────────────┤
#   │                                                             │
#   │  用户输入                                                    │
#   │     │                                                       │
#   │     ▼                                                       │
#   │  ┌──────────┐    RAG检索     ┌──────────────┐              │
#   │  │ Designer │ ─────────────► │  向量数据库   │              │
#   │  │ 设计师   │ ◄─────────────  │ (100篇小说)  │              │
#   │  └──────────┘                └──────────────┘              │
#   │       │ 大纲                                                │
#   │       ▼                                                     │
#   │  ┌──────────┐   全局设定注入                                │
#   │  │  Writer  │ ◄──────────── 主角档案 + 世界观 + 历史记忆    │
#   │  │  写手    │                                               │
#   │  └──────────┘                                               │
#   │       │ 草稿                                                │
#   │       ▼                                                     │
#   │  ┌──────────┐   全局设定比对                                │
#   │  │  Critic  │ ◄──────────── 检查清单（人设/世界观/禁止元素）│
#   │  │  审核员  │                                               │
#   │  └──────────┘                                               │
#   │       │                                                     │
#   │    PASS? ──── 否 ──► 打回 Writer 重写（最多3轮）            │
#   │       │ 是                                                  │
#   │       ▼                                                     │
#   │    保存章节 → 结束                                          │
#   │                                                             │
#   └─────────────────────────────────────────────────────────────┘
#
# 【技术栈说明】
# - LangGraph：多 Agent 工作流编排框架（核心）
# - LangChain：LLM 调用、Prompt 管理、Tool 定义
# - Chroma：本地向量数据库（RAG）
# - 智谱 GLM-4-Flash：大语言模型（免费、速度快）
#
# 【为什么选择 LangGraph 而不是其他框架？】
# 对比：
#   - LangChain Chains：线性流程，不支持循环和条件分支
#   - AutoGen：多 Agent 对话模式，适合开放式任务
#   - CrewAI：角色扮演模式，适合团队协作场景
#   - LangGraph：状态机模式，支持循环、条件分支、状态持久化
#
# 小说创作需要"反思循环"（写 → 审核 → 改 → 再审核），
# 这正是 LangGraph 的强项。
#
# 【类比自动化测试】
# LangGraph 的工作流就像你的 CI/CD Pipeline：
#   - 节点（Node）= 流水线中的每个 Stage（Build/Test/Deploy）
#   - 边（Edge）= Stage 之间的依赖关系
#   - 条件边（Conditional Edge）= 根据测试结果决定是否继续部署
#   - 状态（State）= Pipeline 中传递的上下文（如 artifact、环境变量）
# =============================================================================

import operator
import os
import sys
from typing import Annotated, Sequence, TypedDict, Literal, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 导入我们自己的模块
from config import app_config
from knowledge_base import RAGManager
from agents import (
    create_designer_agent,
    create_writer_agent,
    create_critic_agent,
    create_review_router
)
from tools import save_chapter_tool, word_count_tool


# =============================================================================
# 第一部分：定义全局共享状态（State）
# =============================================================================
#
# 【State 是多 Agent 协作的核心！】
#
# 在 LangGraph 中，所有 Agent 共享同一个 State（状态字典）。
# 每个 Agent 执行完后，返回需要更新的字段，LangGraph 自动合并。
#
# 【State 的设计原则】
# 1. 只存"需要跨 Agent 传递"的信息
# 2. 使用 TypedDict 定义类型，获得 IDE 类型提示
# 3. messages 字段使用 Annotated[..., operator.add] 实现"追加"语义
#    （其他字段默认是"覆盖"语义）
#
# 【与原版 main.py 的对比】
# 原版 State 只有 5 个字段，新版扩展到 10 个字段：
#   新增：current_chapter（当前章节）
#   新增：revision_round（修改轮次，防止死循环）
#   新增：review_passed（审核是否通过，用于路由决策）
#   新增：chapter_summaries（章节摘要列表，实现"记忆"）
#   新增：key_events（关键事件列表，防止情节矛盾）
#
# 【类比自动化测试】
# State 就像你的测试上下文（Test Context）：
#   - 测试开始时初始化
#   - 每个测试步骤读取和修改上下文
#   - 测试结束时包含所有步骤的结果
# =============================================================================

class NovelState(TypedDict):
    # ---- 基础字段（原版已有）----
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 消息历史（追加模式）
    novel_draft: str          # 当前章节草稿
    outline: str              # 故事大纲
    review_feedback: str      # 审核意见
    current_step: str         # 当前执行步骤（用于调试）

    # ---- 新增字段（第二阶段：记忆 + 防幻觉）----
    current_chapter: int      # 当前写到第几章
    revision_round: int       # 当前章节已修改几轮（防止死循环）
    review_passed: bool       # 审核是否通过（路由决策依据）

    # ---- 记忆字段（解决人设崩塌的关键）----
    chapter_summaries: List[str]   # 已完成章节的摘要列表
    key_events: List[str]          # 关键情节节点列表


# =============================================================================
# 第二部分：初始化所有依赖
# =============================================================================

def initialize_system():
    """
    系统初始化函数。

    【为什么要把初始化逻辑单独提取？】
    1. 职责分离：初始化逻辑和业务逻辑分开
    2. 可测试性：可以单独测试初始化过程
    3. 错误处理：初始化失败时给出清晰的错误信息

    Returns:
        tuple: (llm, rag_manager) 初始化好的依赖实例
    """
    print("\n" + "🚀 " * 20)
    print("🚀  NovelAgent 小说创作多智能体系统 启动中...")
    print("🚀 " * 20)

    # ------------------------------------------------------------------
    # 初始化大语言模型
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🤖 [系统初始化] 第1步：初始化大语言模型")
    print("=" * 60)
    print(f"   模型：{app_config.llm.model_name}")
    print(f"   API地址：{app_config.llm.base_url}")
    print(f"   温度参数：{app_config.llm.temperature}")
    print(f"   最大Token：{app_config.llm.max_tokens}")

    if not app_config.llm.api_key:
        print("\n❌ 错误：未找到 API Key！")
        print("   请设置环境变量：export GLM_API_KEY='your_api_key'")
        print("   或在 config.py 中直接设置 api_key")
        sys.exit(1)

    llm = ChatOpenAI(
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        model=app_config.llm.model_name,
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
    )
    print("   ✅ 大语言模型初始化成功")

    # ------------------------------------------------------------------
    # 初始化 RAG 知识库
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📚 [系统初始化] 第2步：初始化 RAG 知识库")
    print("=" * 60)
    print("   💡 RAG（检索增强生成）让设计师在构思时能参考真实小说素材")
    print("   💡 原理：小说文本 → 向量化 → 存入Chroma → 语义检索")

    rag_manager = RAGManager(app_config)
    rag_success = rag_manager.initialize()

    if rag_success:
        status = rag_manager.get_status()
        print(f"   ✅ RAG 知识库初始化成功")
        print(f"   📊 向量数量：{status.get('vector_count', '未知')}")
        print(f"   🧠 嵌入模型：{status.get('embedding_type', '未知')}")
    else:
        print("   ⚠️  RAG 初始化失败，系统将在无 RAG 模式下运行")
        print("   💡 提示：将小说 .txt 文件放入 ./novels/ 目录可启用 RAG")

    return llm, rag_manager


# =============================================================================
# 第三部分：构建 LangGraph 工作流
# =============================================================================

def build_workflow(llm, rag_manager):
    """
    构建 LangGraph 工作流图。

    【LangGraph 工作流的核心概念】

    1. StateGraph：有状态的图，所有节点共享同一个 State
    2. Node（节点）：执行具体任务的函数，签名为 (state) -> dict
    3. Edge（边）：节点之间的固定连接（A 执行完一定去 B）
    4. Conditional Edge（条件边）：根据状态动态决定下一个节点
    5. Entry Point（入口）：图的起始节点
    6. END：图的终止节点

    【工作流图结构】

    START
      │
      ▼
    Designer ──────────────────────────────────────────────────────┐
      │                                                            │
      ▼                                                            │
    Writer ◄──────────────────────────────────────────────────────┤
      │                                                            │
      ▼                                                            │
    Critic                                                         │
      │                                                            │
      ├── PASS ──► END                                            │
      │                                                            │
      └── REVISE ──► Writer（反思循环，最多3轮）                  │
                                                                   │
    （未来扩展：多章节时，Writer完成后可以回到Designer规划下一章）┘

    Args:
        llm: 大语言模型实例
        rag_manager: RAG 管理器实例

    Returns:
        编译好的 LangGraph 应用
    """
    print("\n" + "=" * 60)
    print("🔧 [系统初始化] 第3步：构建 LangGraph 工作流")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 创建各个 Agent 节点函数（使用工厂函数注入依赖）
    # ------------------------------------------------------------------
    print("\n   📦 创建 Agent 节点...")

    print("   ├── 🎨 创建设计师 Agent（含 RAG 检索能力）")
    designer_node = create_designer_agent(llm, rag_manager, app_config)

    print("   ├── ✍️  创建写手 Agent（含记忆注入能力）")
    writer_node = create_writer_agent(llm, app_config)

    print("   ├── 🔍 创建审核 Agent（含全局设定比对能力）")
    critic_node = create_critic_agent(llm, app_config)

    print("   └── 🔀 创建路由函数（反思循环决策器）")
    review_router = create_review_router(app_config)

    # ------------------------------------------------------------------
    # 构建 StateGraph
    # ------------------------------------------------------------------
    print("\n   🗺️  构建工作流图...")

    # 创建图，指定状态类型
    # 【为什么要指定状态类型？】
    # LangGraph 需要知道状态的结构，才能正确合并各节点的输出
    workflow = StateGraph(NovelState)

    # 添加节点
    # 节点名称（字符串）是图中的唯一标识，用于定义边的连接关系
    workflow.add_node("Designer", designer_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Critic", critic_node)
    print("   ├── ✅ 添加节点：Designer, Writer, Critic")

    # 设置入口节点
    workflow.set_entry_point("Designer")
    print("   ├── ✅ 设置入口：Designer")

    # 添加固定边（无条件跳转）
    workflow.add_edge("Designer", "Writer")   # 设计师完成 → 写手开始
    workflow.add_edge("Writer", "Critic")     # 写手完成 → 审核开始
    print("   ├── ✅ 添加固定边：Designer→Writer→Critic")

    # 添加条件边（反思循环的核心！）
    # 【条件边的工作原理】
    # 1. Critic 节点执行完毕
    # 2. LangGraph 调用 review_router(state) 获取路由结果
    # 3. 根据路由结果（"writer" 或 "end"）决定下一个节点
    workflow.add_conditional_edges(
        "Critic",           # 从 Critic 节点出发
        review_router,      # 路由函数（决策器）
        {
            "writer": "Writer",  # 路由结果 "writer" → 跳转到 Writer 节点
            "end": END           # 路由结果 "end" → 跳转到 END（终止）
        }
    )
    print("   ├── ✅ 添加条件边：Critic → (Writer | END)")
    print("   │      PASS → END（审核通过，结束）")
    print("   │      REVISE → Writer（审核未通过，打回重写）")

    # 编译图
    # 【编译做了什么？】
    # 1. 验证图的结构（检查是否有孤立节点、循环引用等问题）
    # 2. 生成执行计划
    # 3. 返回可执行的 CompiledGraph 对象
    app = workflow.compile()
    print("   └── ✅ 工作流图编译成功！")

    # 打印工作流摘要
    print("\n   📊 工作流摘要：")
    print(f"      节点数：3（Designer, Writer, Critic）")
    print(f"      固定边：2（Designer→Writer, Writer→Critic）")
    print(f"      条件边：1（Critic→Writer|END）")
    print(f"      最大修改轮次：{app_config.novel.max_review_rounds}")

    return app


# =============================================================================
# 第四部分：保存最终结果
# =============================================================================

def save_final_result(final_state: dict):
    """
    保存最终创作结果到文件。

    Args:
        final_state: 工作流执行完毕后的最终状态
    """
    print("\n" + "=" * 60)
    print("💾 [结果保存] 保存创作成果")
    print("=" * 60)

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存大纲
    outline = final_state.get("outline", "")
    if outline:
        outline_path = os.path.join(output_dir, "outline.md")
        with open(outline_path, "w", encoding="utf-8") as f:
            f.write(f"# {app_config.novel.title}\n\n## 故事大纲\n\n{outline}")
        print(f"   ✅ 大纲已保存：{outline_path}")

    # 保存章节正文
    novel_draft = final_state.get("novel_draft", "")
    current_chapter = final_state.get("current_chapter", 1)
    if novel_draft:
        chapter_path = os.path.join(output_dir, f"chapter_{current_chapter:02d}.txt")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(f"# 第{current_chapter}章\n\n{novel_draft}")
        print(f"   ✅ 第{current_chapter}章已保存：{chapter_path}")

        # 使用工具统计字数
        word_stats = word_count_tool.invoke({"text": novel_draft})
        print(f"   📊 {word_stats}")

    # 保存审核记录
    review_feedback = final_state.get("review_feedback", "")
    if review_feedback:
        review_path = os.path.join(output_dir, f"review_chapter_{current_chapter:02d}.txt")
        with open(review_path, "w", encoding="utf-8") as f:
            f.write(f"# 第{current_chapter}章审核记录\n\n{review_feedback}")
        print(f"   ✅ 审核记录已保存：{review_path}")

    # 保存完整状态摘要
    summary_path = os.path.join(output_dir, "session_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"创作会话摘要\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"小说标题：{app_config.novel.title}\n")
        f.write(f"当前章节：第{current_chapter}章\n")
        f.write(f"修改轮次：{final_state.get('revision_round', 0)}\n")
        f.write(f"审核结果：{'通过' if final_state.get('review_passed') else '未通过'}\n")
        f.write(f"关键事件数：{len(final_state.get('key_events', []))}\n")
        f.write(f"\n关键事件列表：\n")
        for event in final_state.get("key_events", []):
            f.write(f"  • {event}\n")
    print(f"   ✅ 会话摘要已保存：{summary_path}")


# =============================================================================
# 第五部分：主函数
# =============================================================================

def main():
    """
    主函数：系统入口。

    【完整执行流程】
    1. 系统初始化（LLM + RAG）
    2. 构建 LangGraph 工作流
    3. 设置初始状态
    4. 流式执行工作流（stream 模式）
    5. 实时打印每个节点的输出
    6. 保存最终结果
    """

    # ------------------------------------------------------------------
    # 阶段1：系统初始化
    # ------------------------------------------------------------------
    llm, rag_manager = initialize_system()

    # ------------------------------------------------------------------
    # 阶段2：构建工作流
    # ------------------------------------------------------------------
    app = build_workflow(llm, rag_manager)

    # ------------------------------------------------------------------
    # 阶段3：设置初始状态
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📋 [执行准备] 设置初始状态")
    print("=" * 60)

    # 用户的初始想法（这是整个创作流程的"种子"）
    user_idea = (
        "我想写一个关于一个测试开发工程师，"
        "通过自己的努力顺利成为Agent开发工程师的故事。"
        "主角叫陈默，28岁，内敛执着，有点完美主义。"
        "故事要有真实的技术细节，展现转型过程中的挫折和成长。"
    )

    print(f"   💡 用户想法：{user_idea[:80]}...")

    # 【初始状态设计说明】
    # 所有字段都需要初始化，即使是空值，
    # 因为 TypedDict 要求所有字段都存在。
    initial_state: NovelState = {
        "messages": [HumanMessage(content=user_idea)],
        "novel_draft": "",
        "outline": "",
        "review_feedback": "",
        "current_step": "init",
        "current_chapter": 1,
        "revision_round": 0,
        "review_passed": False,
        "chapter_summaries": [],
        "key_events": [],
    }

    print(f"   ✅ 初始状态设置完成")
    print(f"   📖 目标：第 {initial_state['current_chapter']} 章")
    print(f"   🔄 最大修改轮次：{app_config.novel.max_review_rounds}")

    # ------------------------------------------------------------------
    # 阶段4：执行工作流（流式模式）
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🎬 [开始执行] 启动多智能体创作流水线")
    print("=" * 60)
    print()
    print("   工作流：Designer → Writer → Critic → (Writer → Critic)* → END")
    print("   说明：括号内为反思循环，最多执行 3 次")
    print()

    # 【stream vs invoke 的区别】
    # invoke：等待整个工作流执行完毕，一次性返回最终状态
    # stream：每个节点执行完毕后立即 yield 一个事件，实时返回
    #
    # 为什么用 stream？
    # 1. 实时反馈：用户可以看到每个 Agent 的执行进度
    # 2. 调试友好：可以看到每个节点的输出，便于排查问题
    # 3. 用户体验：不需要等待整个流程结束才看到结果
    #
    # 【类比自动化测试】
    # stream 就像你的测试框架的"实时日志输出"，
    # 而不是等所有测试跑完才看报告。

    final_state = None
    node_execution_count = {}  # 记录每个节点执行了几次

    print("=" * 60)
    print("📡 开始流式执行（每个节点完成后实时输出）")
    print("=" * 60)

    for event in app.stream(initial_state):
        # event 是一个字典：{节点名称: 节点输出}
        # 每次 yield 只包含一个节点的输出
        for node_name, node_output in event.items():

            # 统计节点执行次数
            node_execution_count[node_name] = node_execution_count.get(node_name, 0) + 1
            exec_count = node_execution_count[node_name]

            print(f"\n{'━' * 60}")
            print(f"📍 节点执行完毕：【{node_name}】（第 {exec_count} 次执行）")
            print(f"{'━' * 60}")

            # 根据节点类型打印关键输出
            if node_name == "Designer":
                outline = node_output.get("outline", "")
                if outline:
                    print(f"\n📋 大纲摘要（前500字）：")
                    print(outline[:500] + "..." if len(outline) > 500 else outline)

            elif node_name == "Writer":
                draft = node_output.get("novel_draft", "")
                if draft:
                    print(f"\n📝 正文摘要（前400字）：")
                    print(draft[:400] + "..." if len(draft) > 400 else draft)

                summaries = node_output.get("chapter_summaries", [])
                events = node_output.get("key_events", [])
                if summaries or events:
                    print(f"\n🧠 记忆更新：{len(summaries)} 章摘要，{len(events)} 个关键事件")

            elif node_name == "Critic":
                feedback = node_output.get("review_feedback", "")
                passed = node_output.get("review_passed", False)
                revision = node_output.get("revision_round", 0)

                print(f"\n🔍 审核结论：{'✅ PASS（通过）' if passed else '❌ REVISE（需修改）'}")
                print(f"   修改轮次：{revision}")
                if feedback:
                    print(f"\n   审核详情（前300字）：")
                    print(f"   {feedback[:300]}{'...' if len(feedback) > 300 else ''}")

            # 保存最新状态（用于最终保存）
            final_state = node_output

    # ------------------------------------------------------------------
    # 阶段5：执行完毕，汇总结果
    # ------------------------------------------------------------------
    print("\n" + "🎉 " * 20)
    print("🎉  创作流程全部结束！")
    print("🎉 " * 20)

    print("\n" + "=" * 60)
    print("📊 执行统计")
    print("=" * 60)
    for node_name, count in node_execution_count.items():
        print(f"   {node_name}：执行了 {count} 次")

    total_revisions = node_execution_count.get("Writer", 1) - 1
    print(f"\n   总修改轮次：{total_revisions}")
    print(f"   审核通过：{'是' if final_state and final_state.get('review_passed') else '否（已达最大轮次）'}")

    # ------------------------------------------------------------------
    # 阶段6：保存结果
    # ------------------------------------------------------------------
    if final_state:
        # 合并最终状态（stream 模式下 final_state 只是最后一个节点的输出）
        # 需要重新 invoke 一次获取完整状态，或者在 stream 过程中累积状态
        # 这里我们用 invoke 获取完整的最终状态
        print("\n" + "=" * 60)
        print("🔄 获取完整最终状态...")
        print("=" * 60)

        try:
            complete_final_state = app.invoke(initial_state)
            save_final_result(complete_final_state)
        except Exception as e:
            print(f"   ⚠️  获取完整状态失败：{e}")
            print("   💡 使用流式执行中的最后一个节点输出作为结果")
            save_final_result(final_state)

    print("\n" + "=" * 60)
    print("✨ 感谢使用 NovelAgent！")
    print("=" * 60)
    print(f"   📁 输出文件保存在：./output/")
    print(f"   📖 小说标题：{app_config.novel.title}")
    print(f"   👤 主角：{app_config.character.name}")
    print()
    print("   【学习总结】")
    print("   ✅ 第一阶段：设计-撰写-审核 闭环（LangGraph 反思循环）")
    print("   ✅ 第二阶段：全局设定注入（防止人设崩塌）+ 记忆系统（防止情节矛盾）")
    print("   ✅ 第三阶段：RAG 检索增强（设计师参考真实素材构思大纲）")
    print()
    print("   【下一步学习方向】")
    print("   → 多章节自动续写（在 END 前判断是否继续写下一章）")
    print("   → 引入 Human-in-the-Loop（人工审核节点）")
    print("   → 使用 LangGraph Persistence 实现跨会话记忆")
    print("   → 接入更强的嵌入模型提升 RAG 检索质量")
    print("=" * 60)


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    # 【为什么要有这个判断？】
    # 当直接运行 python main.py 时，__name__ == "__main__"
    # 当被其他模块 import 时，__name__ == "main"（不会执行 main()）
    # 这是 Python 的标准实践，防止模块被导入时意外执行代码
    main()
