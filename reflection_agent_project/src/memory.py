"""
记忆模块
存储智能体的执行与反思轨迹，为迭代优化提供上下文
"""

from typing import List, Dict, Any, Optional


class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。

    负责记录每一次"执行-反思"循环的完整记录，
    并能在构建提示词时将记忆序列化为连贯文本。
    """

    def __init__(self):
        """初始化一个空列表来存储所有记录。"""
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        Args:
            record_type: 记录的类型 ('execution' 或 'reflection')
            content: 记录的具体内容（生成的代码或反思反馈）
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。

        Returns:
            格式化的轨迹文本，包含历次尝试和反馈
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(
                    f"--- 上一轮尝试 (代码) ---\n{record['content']}"
                )
            elif record['type'] == 'reflection':
                trajectory_parts.append(
                    f"--- 评审员反馈 ---\n{record['content']}"
                )

        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次的执行结果（最新生成的代码）。

        Returns:
            最新代码字符串，如果不存在则返回 None
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

    def get_last_reflection(self) -> Optional[str]:
        """
        获取最近一次的反思反馈。

        Returns:
            最新反馈字符串，如果不存在则返回 None
        """
        for record in reversed(self.records):
            if record['type'] == 'reflection':
                return record['content']
        return None

    def clear(self):
        """清空所有记忆记录。"""
        self.records.clear()

    def __repr__(self):
        return f"<Memory records={len(self.records)}>"
