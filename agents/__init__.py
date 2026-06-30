# =============================================================================
# agents/__init__.py —— agents 包的导出入口
# =============================================================================
#
# 【为什么需要 __init__.py？】
# Python 的包机制要求每个目录都有 __init__.py 才能被当作"包"导入。
# 在这里统一导出所有 Agent 的工厂函数，让外部使用更简洁：
#
#   # 不用这样：
#   from agents.designer import create_designer_agent
#   from agents.writer import create_writer_agent
#
#   # 可以这样：
#   from agents import create_designer_agent, create_writer_agent
#
# 【类比自动化测试】
# 就像你的测试框架有一个 conftest.py 统一管理 fixture，
# __init__.py 统一管理模块的公开接口。
# =============================================================================

from agents.designer import create_designer_agent
from agents.writer import create_writer_agent
from agents.critic import create_critic_agent, create_review_router

__all__ = [
    "create_designer_agent",
    "create_writer_agent",
    "create_critic_agent",
    "create_review_router",
]
