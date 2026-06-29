import operator,os
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# 1. 定义全局共享状态 (State)
# 所有的 Agent 都会读取和修改这个状态，这是多智能体协作的核心
class NovelState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 消息历史
    novel_draft: str  # 当前小说草稿
    outline: str  # 当前大纲
    review_feedback: str  # 审核意见
    current_step: str  # 当前处于哪个环节


# 初始化大模型（建议替换为你实际使用的模型）
llm = ChatOpenAI(
    api_key=os.environ.get("GLM_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="glm-4-flash",  # 【修改点3】指定模型名称（免费且速度快）
    temperature=0.7)


# 2. 定义各个智能体节点 (Nodes)

def designer_agent(state: NovelState):
    print("🎨 设计师正在构思大纲...")
    """设计师 Agent：负责构思大纲和剧情"""
    prompt = [
        SystemMessage(content="你是一位资深小说架构师。请根据用户的初步想法，生成一个包含起因、经过、高潮、结局的详细大纲。"),
        HumanMessage(content=f"用户的初步想法：{state['messages'][-1].content}")
    ]
    response = llm.invoke(prompt)
    return {"outline": response.content, "messages": [AIMessage(content=response.content, name="Designer")]}


def writer_agent(state: NovelState):
    print("🎨 写手正在扩写正文...")
    """写手 Agent：负责根据大纲扩写正文"""
    prompt = [
        SystemMessage(content="你是一位文笔极佳的小说家。请根据提供的大纲扩写正文，注意情感细腻和细节描写。"),
        HumanMessage(content=f"大纲：{state['outline']}\n\n请开始撰写第一章。")
    ]
    response = llm.invoke(prompt)
    return {"novel_draft": response.content, "messages": [AIMessage(content=response.content, name="Writer")]}


def critic_agent(state: NovelState):
    """审核 Agent：负责逻辑校验和反思反馈"""
    prompt = [
        SystemMessage(
            content="你是一位严苛的小说主编。请审查以下草稿，检查是否有逻辑漏洞、人设崩塌或剧情拖沓。如果完美，请回复'PASS'；否则给出具体的修改意见。"),
        HumanMessage(content=f"大纲：{state['outline']}\n\n草稿：{state['novel_draft']}")
    ]
    response = llm.invoke(prompt)
    return {"review_feedback": response.content, "messages": [AIMessage(content=response.content, name="Critic")]}


# 3. 定义路由逻辑 (Routing / Conditional Edges)
# 这是反思模式的核心：决定是结束流程，还是打回重写

def review_router(state: NovelState) -> Literal["writer", "end"]:
    """根据审核 Agent 的反馈决定下一步"""
    feedback = state["review_feedback"]
    if "PASS" in feedback.upper():
        return "end"  # 审核通过，结束流程
    else:
        return "writer"  # 审核未通过，打回给写手重写


# 4. 组装工作流图 (Graph)

workflow = StateGraph(NovelState)

# 添加节点
workflow.add_node("Designer", designer_agent)
workflow.add_node("Writer", writer_agent)
workflow.add_node("Critic", critic_agent)

# 定义边 (Edges)
workflow.set_entry_point("Designer")  # 入口：设计师先构思
workflow.add_edge("Designer", "Writer")  # 设计师 -> 写手
workflow.add_edge("Writer", "Critic")  # 写手 -> 审核员
workflow.add_conditional_edges(  # 审核员 -> 条件分支（反思循环）
    "Critic",
    review_router,
    {
        "writer": "Writer",  # 未通过，回到写手
        "end": END  # 通过，结束
    }
)

# 编译图
app = workflow.compile()

# 5. 运行测试
if __name__ == "__main__":
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content="我想写一个关于一个测试开发工程师，通过自己的努力顺利成为Agent开发工程师的故事。")],
        "novel_draft": "",
        "outline": "",
        "review_feedback": "",
        "current_step": "init"
    }
    print("🚀 小说创作系统启动！正在构建工作流...")
    # 2. 使用 stream 方法,流式方法 代替 invoke
    # stream 会返回一个生成器，每次大模型产生一个事件就会 yield 一次
    for event in app.stream(initial_state):
    # event是字典，其中key是节点名称，value是该节点的输出
        for node_name,node_output in event.items():
            print(f"\n ---- 节点[{node_name}]执行完毕！  ----")
            # # 3. 实时打印当前节点生成的内容
            if "novel_draft" in node_output and node_output["novel_draft"]:
                print(node_output["novel_draft"])
    #             如果节点输出message 也打印出来
            if "messages" in node_output and node_output["messages"]:
                last_message = node_output["messages"][-1]
                print(f"{last_message.content}")

    print("\n🎉 创作流程全部结束！")

    # 执行流水线
    # final_state = app.invoke(initial_state)
    #
    # print("=== 最终小说草稿 ===")
    # print(final_state["novel_draft"])