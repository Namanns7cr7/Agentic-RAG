from typing import Any, Dict
from datetime import datetime

class CalculatorTool:
    name = "calculator"
    description = "Evaluate a basic Python arithmetic expression, e.g. '2*(3+4)'."

    def run(self, expression: str) -> str:
        allowed = {k: v for k, v in vars(__builtins__).items() if k in ('abs','round')}
        try:
            result = eval(expression, {'__builtins__': allowed}, {})
            return str(result)
        except Exception as e:
            return f"Calculator error: {e}"

class TimeTool:
    name = "time"
    description = "Return current date-time as ISO string."
    def run(self, _: str = '') -> str:
        return datetime.now().isoformat(timespec='seconds')
