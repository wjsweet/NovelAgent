# =============================================================================
# config.py —— 全局配置中心
# =============================================================================
#
# 【为什么要有这个文件？】
# 在 Agent 开发中，"配置与代码分离" 是最重要的原则之一。
# 把所有"会变化的东西"（模型参数、角色设定、世界观）集中在一个地方，
# 好处是：
#   1. 修改设定时只需改这一个文件，不用翻遍所有 Agent 代码
#   2. 多个 Agent 共享同一份设定，保证"人设一致性"
#   3. 便于版本管理和团队协作
#
# 【类比自动化测试】
# 就像你在做自动化测试时会有一个 config.yaml 或 conftest.py，
# 把测试环境、账号密码、超时时间都放在里面，而不是硬编码在每个测试用例里。
# =============================================================================

import os
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# 第一部分：大模型配置
# =============================================================================

@dataclass
class LLMConfig:
    """
    大模型连接配置。
    使用 dataclass 而不是普通字典，好处是有类型提示、IDE 自动补全、更易读。
    """
    api_key: str = field(default_factory=lambda: os.environ.get("GLM_API_KEY", ""))
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    model_name: str = "glm-4-flash"       # 主力模型（速度快、免费）
    temperature: float = 0.7              # 创意度：0=保守，1=天马行空
    max_tokens: int = 4096                # 单次最大输出 token 数

    # 嵌入模型配置（用于 RAG 向量化）
    embedding_model: str = "embedding-3"  # 智谱的嵌入模型


# =============================================================================
# 第二部分：小说全局设定（"灵魂注入"）
# =============================================================================
#
# 【为什么需要全局设定？解决"幻觉"和"人设崩塌"问题】
#
# 大模型的"幻觉"问题在小说创作中表现为：
#   - 第1章主角叫"李明"，第3章变成了"李亮"
#   - 第1章主角是内向程序员，第5章突然变成了外向社交达人
#   - 世界观设定是"近未来科技"，但突然出现了魔法
#
# 解决方案：把这些"不变的事实"作为 SystemMessage 注入每个 Agent，
# 让每个 Agent 在生成内容前都"对照设定检查"。
#
# 【类比自动化测试】
# 就像你的测试用例有"前置条件"（precondition），
# 全局设定就是所有 Agent 的"前置条件"，必须满足才能继续。
# =============================================================================

@dataclass
class CharacterProfile:
    """主角人物档案"""
    name: str = "陈默"                    # 主角姓名（全程不能变！）
    age: int = 28
    gender: str = "男"
    occupation_start: str = "自动化测试开发工程师"   # 起点职业
    occupation_end: str = "Agent 应用开发工程师"     # 目标职业
    personality: str = (
        "内敛、执着、有点完美主义。"
        "遇到问题喜欢刨根问底，不搞清楚原理不罢休。"
        "表面冷静，内心对技术充满热情。"
        "不善言辞，但代码写得极其优雅。"
    )
    background: str = (
        "985 计算机本科毕业，在一家中型互联网公司做了 4 年自动化测试。"
        "精通 Python、Selenium、pytest，对 CI/CD 流程了如指掌。"
        "某天偶然接触到 LangChain，被 Agent 的智能性深深震撼，"
        "决定利用业余时间转型，用 6 个月完成蜕变。"
    )
    growth_arc: str = (
        "从'测试思维'（验证、断言、边界条件）"
        "逐渐融合'Agent思维'（规划、工具调用、反思循环），"
        "最终发现两者本质相通：都是在构建可靠的自动化系统。"
    )


@dataclass
class WorldSetting:
    """世界观设定"""
    era: str = "2024-2025年，AI大爆发时代"
    location: str = "中国一线城市，某互联网公司"
    tech_background: str = (
        "ChatGPT 引发 AI 革命，各大公司疯狂招募 AI 工程师。"
        "LangChain、LangGraph、AutoGen 等 Agent 框架百花齐放。"
        "传统测试岗位面临 AI 冲击，但懂 AI 的测试工程师反而更值钱。"
    )
    tone: str = "现实主义 + 励志成长，有技术深度，避免玛丽苏爽文套路"
    forbidden_elements: List[str] = field(default_factory=lambda: [
        "穿越、重生、系统金手指",
        "一夜暴富、天才光环",
        "感情线喧宾夺主",
        "技术描写不符合实际"
    ])


@dataclass
class NovelMeta:
    """小说元数据"""
    title: str = "从测试到智能：一个工程师的 Agent 转型之路"
    genre: str = "职场成长 / 技术流"
    target_chapters: int = 10            # 目标章节数
    words_per_chapter: int = 2000        # 每章目标字数
    current_chapter: int = 1             # 当前写到第几章
    max_review_rounds: int = 3           # 最多审核几轮（防止死循环）


# =============================================================================
# 第三部分：RAG 配置
# =============================================================================
#
# 【什么是 RAG？为什么需要它？】
#
# RAG = Retrieval-Augmented Generation（检索增强生成）
# 问题：大模型的知识是训练时"冻结"的，它不知道你的100篇参考小说写了什么。
# 解决：先把100篇小说"向量化"存入数据库，Agent 构思时先"检索"相关段落，
#       再把检索结果塞进 Prompt，让模型"参考"这些内容生成。
#
# 【类比自动化测试】
# 就像你在写测试用例前，先去查"测试用例库"看有没有类似的用例可以复用，
# RAG 就是让 Agent 在生成前先"查资料"。
# =============================================================================

@dataclass
class RAGConfig:
    """RAG 向量数据库配置"""
    # 向量数据库存储路径（本地持久化）
    vector_db_path: str = "./vector_store"

    # 小说素材目录（把你的100篇小说放在这里）
    novels_dir: str = "./novels"

    # 每次检索返回的最相关片段数量
    top_k: int = 3

    # 文本分块大小（把长文本切成小块，便于检索）
    chunk_size: int = 500

    # 相邻块的重叠字符数（防止语义在块边界断裂）
    chunk_overlap: int = 50

    # 相似度阈值（低于此值的检索结果会被过滤）
    similarity_threshold: float = 0.5


# =============================================================================
# 第四部分：统一配置入口
# =============================================================================

class AppConfig:
    """
    应用总配置类。
    所有模块通过 `from config import app_config` 获取配置，
    保证全局只有一份配置实例（单例模式）。
    """
    def __init__(self):
        self.llm = LLMConfig()
        self.character = CharacterProfile()
        self.world = WorldSetting()
        self.novel = NovelMeta()
        self.rag = RAGConfig()

    def get_global_setting_prompt(self) -> str:
        """
        生成"全局设定提示词"，注入每个 Agent 的 SystemMessage。

        【核心设计思路】
        把所有"不变的事实"转成自然语言，作为每个 Agent 的"宪法"。
        Agent 在生成任何内容前，都必须先"读宪法"。
        这是解决人设崩塌最直接有效的方法。
        """
        c = self.character
        w = self.world
        n = self.novel

        forbidden = "\n".join([f"  - {item}" for item in w.forbidden_elements])

        return f"""
【小说全局设定 - 所有内容必须严格遵守】

📖 小说基本信息：
  - 标题：{n.title}
  - 类型：{n.genre}
  - 基调：{w.tone}

👤 主角档案（人设不可崩塌！）：
  - 姓名：{c.name}（全文唯一，不得更改）
  - 年龄：{c.age}岁
  - 性别：{c.gender}
  - 起点职业：{c.occupation_start}
  - 目标职业：{c.occupation_end}
  - 性格特征：{c.personality}
  - 人物背景：{c.background}
  - 成长弧线：{c.growth_arc}

🌍 世界观设定：
  - 时代背景：{w.era}
  - 故事地点：{w.location}
  - 技术背景：{w.tech_background}

🚫 严格禁止出现的元素：
{forbidden}

⚠️ 重要提醒：以上设定是"宪法级"约束，任何生成内容不得违反。
""".strip()


# 全局单例，所有模块直接 import 使用
app_config = AppConfig()


# =============================================================================
# 调试入口：直接运行此文件可查看配置内容
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📋 当前全局设定预览")
    print("=" * 60)
    print(app_config.get_global_setting_prompt())
    print("\n" + "=" * 60)
    print(f"🤖 使用模型：{app_config.llm.model_name}")
    print(f"📚 RAG 素材目录：{app_config.rag.novels_dir}")
    print(f"🗄️  向量库路径：{app_config.rag.vector_db_path}")
    print("=" * 60)
