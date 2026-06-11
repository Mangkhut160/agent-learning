"""
CalculatorTool - 数学计算工具
=============================

内置的数学计算工具，支持基本运算和常见数学函数。
使用Python的ast模块安全地解析和执行数学表达式。

支持的运算:
- 加法 (+)
- 减法 (-)
- 乘法 (*)
- 除法 (/)
- 乘方 (**)
- 平方根 (sqrt)
- 圆周率 (pi)
- 自然常数 (e)

使用示例:
    from hello_agents.tools.builtin import CalculatorTool

    calc = CalculatorTool()
    result = calc.run({"expression": "2 + 3 * 4"})
    print(result)  # 14

    result = calc.run({"expression": "sqrt(16) + pi"})
    print(result)  # 19.14159...
"""

import ast
import operator
import math
from typing import Dict, Any, List

from ..base import Tool, ToolParameter


class CalculatorTool(Tool):
    """
    数学计算工具

    支持基本四则运算和常用数学函数。采用AST（抽象语法树）解析
    表达式，比eval()更安全，可以避免任意代码执行风险。

    继承自:
        Tool: 工具基类

    Example:
        >>> calc = CalculatorTool()
        >>> calc.run({"expression": "2+3"})
        '5'
    """

    def __init__(self):
        """
        初始化计算器工具
        """
        super().__init__(
            name="calculator",
            description="数学计算工具，支持基本运算(+,-,*,/,**)、比较运算和数学函数(sqrt,pi)"
        )

        # 定义支持的运算符映射
        self._operators = {
            ast.Add: operator.add,      # +
            ast.Sub: operator.sub,      # -
            ast.Mult: operator.mul,     # *
            ast.Div: operator.truediv,  # /
            ast.Pow: operator.pow,      # **
        }

        # 定义支持的数学函数
        self._functions = {
            'sqrt': math.sqrt,
            'pi': lambda: math.pi,
            'e': lambda: math.e,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log10,
            'ln': math.log,
            'abs': abs,
            'round': round,
        }

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行数学计算

        Args:
            parameters: 包含 "expression" 键的参数字典

        Returns:
            计算结果的字符串表示，如果计算失败返回错误信息
        """
        expression = parameters.get("expression", "")

        if not expression or not expression.strip():
            return "❌ 错误: 计算表达式不能为空"

        try:
            result = self._evaluate(expression)
            return str(result)
        except ZeroDivisionError:
            return "❌ 错误: 除数不能为零"
        except Exception as e:
            return f"❌ 计算失败: {str(e)}，请检查表达式格式是否正确"

    def get_parameters(self) -> List[ToolParameter]:
        """
        获取工具参数定义

        Returns:
            包含一个参数定义的列表
        """
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式，支持 +,-,*,/,**,sqrt 等。例如: 2+3*4, sqrt(16)+pi"
            )
        ]

    def _evaluate(self, expression: str) -> float:
        """
        安全地评估数学表达式

        使用AST解析表达式，只允许数学运算，拒绝任意代码执行。

        Args:
            expression: 数学表达式字符串

        Returns:
            计算结果（浮点数）
        """
        # 预处理表达式
        expression = expression.strip()

        # 解析为AST
        node = ast.parse(expression, mode='eval')

        # 递归求值
        return self._eval_node(node.body)

    def _eval_node(self, node) -> Any:
        """
        递归评估AST节点

        Args:
            node: AST节点

        Returns:
            求值结果
        """
        # 常数值
        if isinstance(node, ast.Constant):
            # Python 3.8+ 使用Constant，之前的版本可能使用Num
            return node.value

        # 二元运算 (a + b, a - b, etc.)
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op)}")
            return op(left, right)

        # 一元运算 (-a)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return self._eval_node(node.operand)

        # 函数调用 (sqrt(x), pi, etc.)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self._functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self._functions[func_name](*args)
            raise ValueError(f"不支持的函数: {func_name}")

        # 名称引用 (pi, e)
        if isinstance(node, ast.Name):
            if node.id in self._functions:
                return self._functions[node.id]()
            raise ValueError(f"不支持的名称: {node.id}")

        raise ValueError(f"不支持的表达式节点: {type(node)}")