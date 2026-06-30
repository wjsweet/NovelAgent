# =============================================================================
# agents/designer.py —— 设计师 Agent（大纲构思 + RAG 检索增强）
# =============================================================================
#
# 【设计师 Agent 的职责】
# 在整个小说创作流水线中，设计师是"第一棒"：
#   1. 接收用户的初始想法
#   2. 去 RAG 知识库检索相关参考素材（借鉴优秀作品的结构）
#   3. 结合全局设定（主角、世界观）生成详细大纲
#
# 【为什么设计师需要 RAG？】
# 没有 RAG 时：设计师只能靠大模型"凭空想象"，容易生成套路化、雷同的大纲
# 有了 RAG 后：设计师先"读"100篇参考小说，借鉴其中的情节结构和叙事技巧，
#              生成的大纲更有参考价值，更符合目标类型的写作规范
#
# 【类比自动化测试】
# 就像你在设计测试用例前，会先查看"测试用例模板库"和"历史用例"，
# 设计师 Agent 在构思大纲前，先查"小说素材库"。
#
# 【Agent 开发核心模式：ReAct 模式】
# ReAct = Reasoning + Acting（推理 + 行动）
# 设计师的工作流：
#   Reason: "用户想写职场成长故事，我需要参考类似题材的优秀作品"
#   Act:    调用 RAG 检索工具
#   Reason: "检索到了3篇相关素材，我来分析其结构特点"
#   Act:    调用 LLM 生成大纲
# =============================================================================

import os
import sys
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 将项目根目录加入路径，确保可以导入其他模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_designer_agent(llm: ChatOpenAI, rag_manager, config):
    """
    工厂函数：创建设计师 Agent 节点函数。

    【为什么用工厂函数而不是直接定义函数？】
    LangGraph 的节点函数签名必须是 (state) -> dict，
    但设计师需要访问 llm、rag_manager、config 等外部依赖。
    工厂函数通过"闭包"把这些依赖"注入"进去，
    返回一个符合 LangGraph 要求的节点函数。

    这是 Python 中"依赖注入"的常见实现方式，
    类比测试框架中的 fixture 注入。

    Args:
        llm: 大语言模型实例
        rag_manager: RAG 管理器实例
        config: 全局配置实例

    Returns:
        符合 LangGraph 节点签名的函数
    """

    def designer_agent(state: dict) -> dict:
        """
        设计师 Agent 节点函数。

        【执行流程】
        Step 1: 打印执行状态（让用户知道进展）
        Step 2: 从状态中提取用户输入
        Step 3: RAG 检索相关素材
        Step 4: 构建包含全局设定 + 参考素材的 Prompt
        Step 5: 调用 LLM 生成大纲
        Step 6: 更新状态并返回

        Args:
            state: LangGraph 共享状态字典

        Returns:
            dict: 需要更新的状态字段
        """
        print("\n" + "=" * 60)
        print("🎨 [设计师Agent] 开始工作")
        print("=" * 60)
        print("📋 职责：根据用户想法 + RAG参考素材，生成详细大纲")
        print("🔄 当前阶段：第一阶段 - 大纲构思")

        # ------------------------------------------------------------------
        # Step 1: 提取用户输入
        # ------------------------------------------------------------------
        print("\n📥 [Step 1] 提取用户输入...")
        user_input = state["messages"][-1].content
        print(f"   用户想法：{user_input[:100]}{'...' if len(user_input) > 100 else ''}")

        # 检查是否有审核反馈（说明这是重写轮次）
        review_feedback = state.get("review_feedback", "")
        current_round = state.get("revision_round", 0)

        if review_feedback and current_round > 0:
            print(f"\n🔄 [Step 1] 检测到审核反馈（第 {current_round} 轮修改）")
            print(f"   审核意见：{review_feedback[:100]}...")
            print("   💡 设计师将根据审核意见调整大纲")

        # ------------------------------------------------------------------
        # Step 2: RAG 检索参考素材
        # ------------------------------------------------------------------
        print("\n🔍 [Step 2] 启动 RAG 检索，寻找参考素材...")
        print("   💡 原理：把用户想法转成向量，在知识库中找最相似的小说片段")

        reference_text = "（暂无参考素材）"

        if rag_manager and rag_manager._is_initialized:
            # 构建检索查询：结合用户输入和小说类型
            search_query = f"{user_input} 职场成长 技术转型 励志"
            print(f"   🔎 检索关键词：{search_query[:60]}...")

            results = rag_manager.search(search_query, top_k=config.rag.top_k)
            reference_text = rag_manager.format_search_results(results)

            if results:
                print(f"   ✅ RAG 检索成功，找到 {len(results)} 个相关片段")
                print("   💡 这些片段将作为'参考资料'注入设计师的 Prompt")
            else:
                print("   📭 RAG 未找到相关素材，设计师将完全依赖自身能力")
        else:
            print("   ⚠️  RAG 未初始化，跳过检索步骤")
            print("   💡 提示：运行 rag_manager.initialize() 可启用 RAG 功能")

        # ------------------------------------------------------------------
        # Step 3: 构建 Prompt（核心！）
        # ------------------------------------------------------------------
        print("\n📝 [Step 3] 构建 Prompt...")
        print("   💡 Prompt 由三部分组成：")
        print("      ① 全局设定（宪法级约束，防止人设崩塌）")
        print("      ② RAG 参考素材（借鉴优秀作品的结构）")
        print("      ③ 具体任务指令（告诉 LLM 要做什么）")

        # 获取全局设定提示词
        global_setting = config.get_global_setting_prompt()

        # 构建系统提示词（System Prompt）
        # 【Prompt 工程最佳实践】
        # System Prompt 定义 Agent 的"角色"和"约束"
        # Human Message 提供具体的"任务"
        system_prompt = f"""你是一位资深小说架构师，专注于职场成长类小说的结构设计。

{global_setting}

【你的工作原则】
1. 严格遵守上述全局设定，主角姓名、性格、背景不得更改
2. 大纲必须包含：起因、发展、高潮、结局四个阶段
3. 每个阶段需要有具体的情节点，不能只有空洞的描述
4. 技术细节要真实可信（如 LangChain、pytest、CI/CD 等）
5. 成长弧线要符合真实的职业转型规律，不能一蹴而就

【大纲格式要求】
请按以下格式输出：
## 故事大纲

### 第一阶段：起因（第1-2章）
- 核心事件：...
- 主角状态：...
- 关键转折：...

### 第二阶段：发展（第3-6章）
- 核心事件：...
- 主角成长：...
- 遇到的挫折：...

### 第三阶段：高潮（第7-9章）
- 核心冲突：...
- 主角突破：...

### 第四阶段：结局（第10章）
- 结局方式：...
- 主题升华：...

### 核心主题
...

### 写作风格建议
...
"""

        # 构建用户消息（Human Message）
        if review_feedback and current_round > 0:
            # 修改轮次：需要根据审核意见调整大纲
            human_message = f"""请根据以下信息重新设计大纲：

【用户原始想法】
{user_input}

【上一版大纲的审核意见】
{review_feedback}

【参考素材】
{reference_text}

请针对审核意见中指出的问题，重新设计一个更完善的大纲。"""
        else:
            # 首次构思
            human_message = f"""请根据以下信息设计详细大纲：

【用户的初步想法】
{user_input}

【参考素材（来自知识库）】
{reference_text}

请结合参考素材中的优秀结构，为这个故事设计一个引人入胜的大纲。"""

        print(f"   ✅ Prompt 构建完成")
        print(f"   📏 System Prompt 长度：{len(system_prompt)} 字符")
        print(f"   📏 Human Message 长度：{len(human_message)} 字符")

        # ------------------------------------------------------------------
        # Step 4: 调用 LLM 生成大纲
        # ------------------------------------------------------------------
        print("\n🤖 [Step 4] 调用大模型生成大纲...")
        print(f"   🔧 使用模型：{config.llm.model_name}")
        print("   ⏳ 等待模型响应（这可能需要10-30秒）...")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]

        try:
            response = llm.invoke(messages)
            outline = response.content
            print(f"   ✅ 大纲生成成功！长度：{len(outline)} 字符")
        except Exception as e:
            print(f"   ❌ LLM 调用失败：{e}")
            outline = f"大纲生成失败：{str(e)}"

        # ------------------------------------------------------------------
        # Step 5: 打印生成结果预览
        # ------------------------------------------------------------------
        print("\n📄 [Step 5] 大纲预览（前200字）：")
        print("-" * 40)
        print(outline[:200] + "..." if len(outline) > 200 else outline)
        print("-" * 40)

        print("\n✅ [设计师Agent] 工作完成！")
        print(f"   📊 输出：大纲（{len(outline)} 字符）")
        print("   ➡️  下一步：写手 Agent 将根据此大纲撰写正文")

        # ------------------------------------------------------------------
        # Step 6: 返回更新的状态
        # ------------------------------------------------------------------
        # 【LangGraph 状态更新机制】
        # 返回的字典只需包含"需要更新的字段"
        # LangGraph 会自动合并到全局状态中
        # messages 字段使用 operator.add，所以是追加而不是覆盖
        return {
            "outline": outline,
            "current_step": "design_complete",
            "messages": [AIMessage(content=f"[设计师] 大纲已生成：\n{outline}", name="Designer")]
        }

    return designer_agent


# =============================================================================
# 调试入口：单独测试设计师 Agent
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 设计师 Agent 独立测试")
    print("=" * 60)

    from config import app_config
    from knowledge_base.rag_manager import RAGManager

    # 初始化依赖
    llm = ChatOpenAI(
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        model=app_config.llm.model_name,
        temperature=app_config.llm.temperature
    )

    rag = RAGManager(app_config)
    rag.initialize()

    # 创建 Agent
    designer = create_designer_agent(llm, rag, app_config)

    # 模拟状态
    test_state = {
        "messages": [HumanMessage(content="我想写一个测试工程师转型 Agent 开发的故事")],
        "novel_draft": "",
        "outline": "",
        "review_feedback": "",
        "current_step": "init",
        "revision_round": 0
    }

    # 执行
    result = designer(test_state)
    print("\n" + "=" * 60)
    print("📋 完整大纲：")
    print("=" * 60)
    print(result["outline"])
