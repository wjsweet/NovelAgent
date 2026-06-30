# =============================================================================
# knowledge_base/rag_manager.py —— RAG 检索增强生成管理器
# =============================================================================
#
# 【RAG 是什么？为什么 Agent 开发必须掌握它？】
#
# 大模型有两个核心局限：
#   1. 知识截止日期：训练数据有截止，不知道最新信息
#   2. 上下文窗口限制：无法把100篇小说全塞进 Prompt
#
# RAG 的解决思路（三步走）：
#   ① 离线阶段（Indexing）：把文档切块 → 向量化 → 存入向量数据库
#   ② 在线阶段（Retrieval）：用户查询 → 向量化 → 在数据库中找最相似的块
#   ③ 生成阶段（Generation）：把检索到的块 + 用户问题一起喂给大模型
#
# 【类比自动化测试】
# 就像你在做接口测试时，不会把所有测试数据硬编码在代码里，
# 而是存在 Excel/数据库里，测试时动态读取。
# RAG 就是让 Agent 动态读取"知识库"，而不是靠模型自身的"记忆"。
#
# 【技术选型说明】
# - 向量数据库：Chroma（本地持久化，无需部署服务器，适合学习和小项目）
# - 嵌入模型：支持智谱 embedding-3 或本地 sentence-transformers
# - 文本分割：LangChain 的 RecursiveCharacterTextSplitter（递归分割，效果最好）
#
# 【为什么选 Chroma 而不是 Pinecone/Weaviate？】
# Chroma 是本地运行的，不需要注册账号、不需要网络，适合学习阶段。
# 生产环境可以无缝切换到 Pinecone（云端托管，高可用）。
# =============================================================================

import os
import glob
import hashlib
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 【依赖说明】
# langchain-chroma：LangChain 官方的 Chroma 集成包
# chromadb：Chroma 向量数据库本体
try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  [RAG] langchain-chroma 未安装，RAG 功能将降级为 Mock 模式")
    print("   安装命令：pip install langchain-chroma chromadb")

# 嵌入模型：优先使用智谱，降级使用本地模型
try:
    from langchain_community.embeddings import ZhipuAIEmbeddings
    ZHIPU_EMBEDDING_AVAILABLE = True
except ImportError:
    ZHIPU_EMBEDDING_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


# =============================================================================
# Mock 嵌入模型（当所有真实嵌入模型都不可用时的降级方案）
# =============================================================================

class MockEmbeddings:
    """
    Mock 嵌入模型，用于开发调试阶段。
    真实项目中必须替换为真实的嵌入模型。

    【为什么需要 Mock？】
    在 Agent 开发中，"分层测试"很重要：
    - 先用 Mock 验证整体流程是否跑通
    - 再换真实模型验证效果
    这和自动化测试中的 Mock/Stub 思想完全一致！
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """把文本列表转成向量列表（Mock：用哈希值生成伪向量）"""
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """把查询文本转成向量（Mock）"""
        return self._text_to_vector(text)

    def _text_to_vector(self, text: str) -> List[float]:
        """用 MD5 哈希生成一个 128 维的伪向量（仅用于测试）"""
        hash_bytes = hashlib.md5(text.encode()).digest()
        # 把 16 字节的哈希扩展成 128 维向量
        vector = []
        for byte in hash_bytes:
            for bit in range(8):
                vector.append(float((byte >> bit) & 1))
        return vector


# =============================================================================
# RAG 管理器核心类
# =============================================================================

class RAGManager:
    """
    RAG 知识库管理器。

    职责：
    1. 加载小说文件（支持 .txt、.md 格式）
    2. 文本分块（Chunking）
    3. 向量化并存入 Chroma
    4. 提供语义检索接口

    【设计模式：门面模式（Facade Pattern）】
    把复杂的 RAG 流程（加载→分块→向量化→存储→检索）封装成简单的接口，
    外部 Agent 只需调用 search() 方法，不需要关心内部实现细节。
    这和你在测试框架中封装 Page Object 是同一个思路。
    """

    def __init__(self, config):
        """
        初始化 RAG 管理器。

        Args:
            config: AppConfig 实例，包含 RAG 相关配置
        """
        self.config = config
        self.rag_cfg = config.rag
        self.llm_cfg = config.llm
        self.vector_store: Optional[object] = None
        self._is_initialized = False

        print("🗄️  [RAG管理器] 初始化中...")
        print(f"   📁 素材目录：{self.rag_cfg.novels_dir}")
        print(f"   💾 向量库路径：{self.rag_cfg.vector_db_path}")
        print(f"   🔢 检索 Top-K：{self.rag_cfg.top_k}")

        # 初始化文本分割器
        # RecursiveCharacterTextSplitter 是目前效果最好的通用分割器
        # 它会按照 ["\n\n", "\n", "。", "，", " ", ""] 的优先级递归分割
        # 保证分割点尽量在语义边界（段落 > 句子 > 词语）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_cfg.chunk_size,
            chunk_overlap=self.rag_cfg.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
            length_function=len,
        )
        print(f"   ✂️  文本分块大小：{self.rag_cfg.chunk_size} 字符，重叠：{self.rag_cfg.chunk_overlap} 字符")

        # 初始化嵌入模型
        self.embeddings = self._init_embeddings()

    def _init_embeddings(self):
        """
        初始化嵌入模型，按优先级降级：
        智谱 embedding-3 > HuggingFace 本地模型 > Mock 模型

        【为什么要有降级策略？】
        在 Agent 开发中，"优雅降级"是非常重要的工程实践。
        当某个依赖不可用时，系统应该用更简单的方式继续运行，
        而不是直接崩溃。这和你在测试中处理"非关键断言"是一个道理。
        """
        api_key = self.llm_cfg.api_key

        if ZHIPU_EMBEDDING_AVAILABLE and api_key:
            try:
                print("   🧠 嵌入模型：智谱 ZhipuAI Embeddings")
                embeddings = ZhipuAIEmbeddings(
                    model=self.llm_cfg.embedding_model,
                    api_key=api_key,
                    zhipuai_api_base="https://open.bigmodel.cn/api/paas/v4/"
                )
                return embeddings
            except Exception as e:
                print(f"   ⚠️  智谱嵌入模型初始化失败：{e}")

        if HUGGINGFACE_AVAILABLE:
            try:
                # 使用中文效果较好的本地嵌入模型
                model_name = "shibing624/text2vec-base-chinese"
                print(f"   🧠 嵌入模型：HuggingFace 本地模型 ({model_name})")
                print("   ⏳ 首次使用需要下载模型，请耐心等待...")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                return embeddings
            except Exception as e:
                print(f"   ⚠️  HuggingFace 嵌入模型初始化失败：{e}")

        print("   🔧 嵌入模型：Mock 模式（仅用于流程测试，检索结果无实际语义）")
        return MockEmbeddings()

    def initialize(self, force_rebuild: bool = False) -> bool:
        """
        初始化向量数据库。

        【核心流程】
        1. 检查向量库是否已存在（避免重复构建）
        2. 如果不存在或强制重建，则加载素材文件
        3. 文本分块 → 向量化 → 存入 Chroma

        Args:
            force_rebuild: 是否强制重建（当素材更新时使用）

        Returns:
            bool: 初始化是否成功
        """
        print("\n" + "=" * 60)
        print("📚 [RAG] 开始初始化知识库...")
        print("=" * 60)

        vector_db_path = self.rag_cfg.vector_db_path

        # 检查是否已有持久化的向量库
        if not force_rebuild and os.path.exists(vector_db_path) and CHROMA_AVAILABLE:
            print(f"✅ [RAG] 发现已有向量库：{vector_db_path}")
            print("   💡 直接加载已有向量库（跳过重建，节省时间）")
            print("   💡 如需重建，请传入 force_rebuild=True")
            try:
                self.vector_store = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=self.embeddings,
                    collection_name="novels"
                )
                count = self.vector_store._collection.count()
                print(f"   📊 向量库中已有 {count} 个文本块")
                self._is_initialized = True
                return True
            except Exception as e:
                print(f"   ⚠️  加载已有向量库失败：{e}，将重新构建")

        # 加载素材文件
        documents = self._load_novels()

        if not documents:
            print("⚠️  [RAG] 未找到任何素材文件，RAG 功能将以 Mock 模式运行")
            print(f"   请将小说 .txt 或 .md 文件放入：{self.rag_cfg.novels_dir}/")
            self._is_initialized = False
            return False

        # 文本分块
        print(f"\n✂️  [RAG] 开始文本分块...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"   📄 原始文档：{len(documents)} 篇")
        print(f"   🧩 分块后：{len(chunks)} 个文本块")
        print(f"   📏 平均块大小：{sum(len(c.page_content) for c in chunks) // len(chunks)} 字符")

        # 向量化并存入数据库
        print(f"\n🔢 [RAG] 开始向量化（这可能需要几分钟）...")

        if CHROMA_AVAILABLE:
            try:
                # 分批处理，避免一次性请求太多导致 API 超时
                batch_size = 50
                total_batches = (len(chunks) + batch_size - 1) // batch_size

                print(f"   📦 共 {total_batches} 批，每批 {batch_size} 个文本块")

                # 先创建空的向量库
                self.vector_store = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=self.embeddings,
                    collection_name="novels"
                )

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    print(f"   ⏳ 处理第 {batch_num}/{total_batches} 批...")
                    self.vector_store.add_documents(batch)

                count = self.vector_store._collection.count()
                print(f"\n✅ [RAG] 向量库构建完成！共 {count} 个向量")
                print(f"   💾 已持久化到：{vector_db_path}")
                self._is_initialized = True
                return True

            except Exception as e:
                print(f"❌ [RAG] 向量化失败：{e}")
                print("   降级为 Mock 模式")
                self._is_initialized = False
                return False
        else:
            print("   ⚠️  Chroma 不可用，使用内存 Mock 存储")
            # Mock 存储：直接保存原始文本块，检索时用关键词匹配
            self._mock_chunks = chunks
            self._is_initialized = True
            return True

    def _load_novels(self) -> List[Document]:
        """
        从素材目录加载小说文件。

        支持格式：.txt、.md
        每个文件会被包装成 LangChain 的 Document 对象，
        Document 包含 page_content（文本内容）和 metadata（元数据）。

        【为什么要保存 metadata？】
        检索时不仅要返回文本内容，还要告诉 Agent "这段话来自哪本书"，
        这样 Agent 可以在生成时注明参考来源，增加可信度。
        """
        novels_dir = self.rag_cfg.novels_dir
        documents = []

        print(f"\n📂 [RAG] 扫描素材目录：{novels_dir}")

        if not os.path.exists(novels_dir):
            print(f"   ⚠️  目录不存在，正在创建：{novels_dir}")
            os.makedirs(novels_dir, exist_ok=True)
            # 创建示例文件，帮助用户了解格式
            self._create_sample_novel(novels_dir)

        # 扫描所有支持的文件格式
        supported_extensions = ["*.txt", "*.md"]
        all_files = []
        for ext in supported_extensions:
            pattern = os.path.join(novels_dir, "**", ext)
            all_files.extend(glob.glob(pattern, recursive=True))

        if not all_files:
            print(f"   ⚠️  目录中没有 .txt 或 .md 文件")
            return documents

        print(f"   📚 发现 {len(all_files)} 个素材文件")

        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()

                if not content:
                    print(f"   ⏭️  跳过空文件：{file_name}")
                    continue

                # 创建 Document 对象，metadata 记录来源信息
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_name,
                        "file_path": file_path,
                        "file_size": file_size,
                        "char_count": len(content)
                    }
                )
                documents.append(doc)
                print(f"   ✅ 加载：{file_name} ({len(content)} 字符)")

            except Exception as e:
                print(f"   ❌ 加载失败：{file_path} - {e}")

        print(f"\n   📊 成功加载 {len(documents)} 篇素材")
        return documents

    def _create_sample_novel(self, novels_dir: str):
        """创建示例小说文件，帮助用户了解素材格式"""
        sample_content = """# 示例小说：程序员的成长之路

第一章：迷茫的起点

李程是一名工作了三年的后端开发工程师。每天面对着相似的需求文档，写着相似的增删改查代码，他开始感到一种深深的倦怠。

"难道我的职业生涯就这样了吗？"他盯着屏幕上密密麻麻的代码，心里涌起一阵茫然。

转机出现在一个普通的周二下午。他在技术论坛上看到了一篇关于 AI Agent 的文章，作者用短短几百行代码，让 AI 自动完成了原本需要人工处理几个小时的数据分析任务。

"这就是未来。"李程心里某个沉睡已久的东西被唤醒了。

第二章：艰难的学习

学习 Agent 开发并不容易。李程发现自己需要补充大量知识：LangChain 的链式调用、向量数据库的原理、Prompt Engineering 的技巧……

每天下班后，他都会在家里学习到深夜。妻子心疼地给他端来热茶，说："别太拼了。"

"再等我三个月。"他抬起头，眼神坚定。

第三章：第一个 Agent

三个月后，李程完成了他的第一个真正意义上的 Agent 系统——一个能自动分析竞品、生成报告的工具。

当他把这个工具演示给团队时，所有人都沉默了。

"这……真的是你一个人做的？"技术总监难以置信地问。

李程微笑着点头。那一刻，所有的熬夜和挫折都值得了。
"""
        sample_path = os.path.join(novels_dir, "sample_growth_story.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        print(f"   📝 已创建示例素材文件：{sample_path}")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, str, float]]:
        """
        语义检索接口。

        【这是 RAG 的核心方法】
        输入：用户的查询文本（自然语言）
        输出：最相关的文本片段列表

        Args:
            query: 检索查询（如"程序员转型的心理描写"）
            top_k: 返回结果数量，默认使用配置值

        Returns:
            List of (content, source, score) tuples
            - content: 文本内容
            - source: 来源文件名
            - score: 相似度分数（越高越相关）
        """
        if top_k is None:
            top_k = self.rag_cfg.top_k

        print(f"\n🔍 [RAG检索] 查询：「{query[:50]}...」" if len(query) > 50 else f"\n🔍 [RAG检索] 查询：「{query}」")
        print(f"   📊 检索 Top-{top_k} 相关片段")

        if not self._is_initialized:
            print("   ⚠️  向量库未初始化，返回空结果")
            return []

        results = []

        if CHROMA_AVAILABLE and self.vector_store is not None:
            try:
                # similarity_search_with_score 返回 (Document, score) 元组列表
                # score 是余弦相似度，范围 [0, 1]，越接近 1 越相似
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=top_k
                )

                for doc, score in docs_with_scores:
                    # Chroma 返回的是距离（越小越相似），转换为相似度
                    similarity = 1 - score if score <= 1 else 1 / (1 + score)

                    if similarity >= self.rag_cfg.similarity_threshold:
                        source = doc.metadata.get("source", "未知来源")
                        results.append((doc.page_content, source, similarity))
                        print(f"   ✅ 找到相关片段 | 来源：{source} | 相似度：{similarity:.3f}")
                        print(f"      内容预览：{doc.page_content[:80]}...")
                    else:
                        print(f"   ⏭️  相似度过低（{similarity:.3f} < {self.rag_cfg.similarity_threshold}），跳过")

            except Exception as e:
                print(f"   ❌ 检索失败：{e}")

        elif hasattr(self, "_mock_chunks"):
            # Mock 模式：简单的关键词匹配
            print("   🔧 使用 Mock 关键词匹配模式")
            query_words = set(query.replace("，", " ").replace("。", " ").split())

            scored_chunks = []
            for chunk in self._mock_chunks:
                content = chunk.page_content
                # 计算关键词命中率作为"相似度"
                hits = sum(1 for word in query_words if word in content and len(word) > 1)
                if hits > 0:
                    score = hits / len(query_words) if query_words else 0
                    scored_chunks.append((content, chunk.metadata.get("source", "未知"), score))

            # 按分数排序，取 top_k
            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            results = scored_chunks[:top_k]

            for content, source, score in results:
                print(f"   ✅ 关键词匹配 | 来源：{source} | 命中率：{score:.3f}")
                print(f"      内容预览：{content[:80]}...")

        if not results:
            print("   📭 未找到相关片段（向量库可能为空或查询无匹配）")

        print(f"   📋 共返回 {len(results)} 个相关片段")
        return results

    def format_search_results(self, results: List[Tuple[str, str, float]]) -> str:
        """
        将检索结果格式化为可注入 Prompt 的文本。

        【Prompt 工程技巧】
        检索结果需要以结构化的方式呈现给大模型，
        让模型清楚地知道"这是参考资料，不是你自己的知识"。
        使用 XML 标签（<reference>）是目前业界常用的做法，
        可以帮助模型区分"参考内容"和"指令内容"。
        """
        if not results:
            return "（暂无相关参考素材）"

        formatted = "【参考素材（来自知识库，请借鉴写作风格和情节处理方式）】\n\n"
        for i, (content, source, score) in enumerate(results, 1):
            formatted += f"--- 参考片段 {i} | 来源：{source} | 相关度：{score:.2f} ---\n"
            formatted += content[:300]  # 限制每个片段的长度，避免 Prompt 过长
            if len(content) > 300:
                formatted += "...(已截断)"
            formatted += "\n\n"

        return formatted.strip()

    def get_status(self) -> dict:
        """返回向量库状态信息，用于调试"""
        status = {
            "initialized": self._is_initialized,
            "chroma_available": CHROMA_AVAILABLE,
            "embedding_type": type(self.embeddings).__name__,
            "vector_db_path": self.rag_cfg.vector_db_path,
            "novels_dir": self.rag_cfg.novels_dir,
        }

        if CHROMA_AVAILABLE and self.vector_store is not None:
            try:
                status["vector_count"] = self.vector_store._collection.count()
            except:
                status["vector_count"] = "未知"
        elif hasattr(self, "_mock_chunks"):
            status["vector_count"] = len(self._mock_chunks)
        else:
            status["vector_count"] = 0

        return status


# =============================================================================
# 调试入口
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import app_config

    print("=" * 60)
    print("🧪 RAG 管理器独立测试")
    print("=" * 60)

    rag = RAGManager(app_config)
    success = rag.initialize()

    print(f"\n📊 向量库状态：{rag.get_status()}")

    if success:
        # 测试检索
        test_queries = [
            "程序员转型的心理挣扎",
            "学习新技术的艰难过程",
            "职场成长励志故事"
        ]
        for query in test_queries:
            results = rag.search(query)
            formatted = rag.format_search_results(results)
            print(f"\n检索结果：\n{formatted}")
            print("-" * 40)
