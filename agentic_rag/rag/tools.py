"""Pluggable tools for the agentic pipeline.

Each tool exposes:
    - name: str
    - description: str
    - run(input_str) -> str
"""

from typing import Any
from datetime import datetime
import ast
import operator
import re
import urllib.request
from html.parser import HTMLParser


class SimpleHTMLTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.in_body = False
        self.ignore_tags = {'script', 'style', 'head', 'meta', 'link', 'noscript'}
        self.current_tag = ""

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.in_body = True
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = ""

    def handle_data(self, data):
        if self.in_body and self.current_tag not in self.ignore_tags:
            t = data.strip()
            if t:
                self.text.append(t)

    def get_text(self):
        return " ".join(self.text)



class CalculatorTool:
    """Safely evaluate arithmetic expressions using AST parsing (no eval)."""

    name = "calculator"
    description = "Evaluate a basic arithmetic expression, e.g. '2*(3+4)'. Supports +, -, *, /, //, %, **."

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _safe_eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op = self._OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(self._safe_eval(node.left), self._safe_eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(self._safe_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def run(self, expression: str) -> str:
        # Extract just the mathematical expression from natural language
        expr = re.sub(r"[^0-9+\-*/().%\s]", "", expression).strip()
        if not expr:
            # Try to find numbers and operators in the original string
            nums = re.findall(r"[\d.]+|[+\-*/()%^]", expression)
            expr = "".join(nums).replace("^", "**")
        if not expr:
            return "Calculator error: no valid arithmetic expression found."
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._safe_eval(tree)
            return str(result)
        except Exception as e:
            return f"Calculator error: {e}"


class TimeTool:
    """Return the current date-time."""

    name = "time"
    description = "Return current date and time as a readable ISO string."

    def run(self, _: str = "") -> str:
        return datetime.now().isoformat(timespec="seconds")


class SummarizeTool:
    """Summarize the retrieved context into key bullet points."""

    name = "summarize"
    description = "Condense a piece of text into 3-5 bullet points."

    def run(self, text: str) -> str:
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        if not sentences:
            return "Nothing to summarize."
        bullets = sentences[:5]
        return "\n".join(f"• {b}" for b in bullets)


class WebScraperTool:
    """Read textual content from a URL."""

    name = "scrape"
    description = "Fetch and extract text content from a given website URL."

    def run(self, input_str: str) -> str:
        # Extract URL from input string
        urls = re.findall(r'(https?://[^\s]+)', input_str)
        if not urls:
            return "Error: No valid HTTP/HTTPS URL provided."
        url = urls[0]
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Agentic RAG Bot)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
            parser = SimpleHTMLTextParser()
            parser.feed(html)
            text = parser.get_text()
            
            # Truncate to avoid context window explosion
            text = text[:1500]
            if not text:
                return "Successfully accessed website, but found no readable text."
            return f"Excerpt from {url}:\n{text}..."
        except Exception as e:
            return f"Failed to scrape {url}: {e}"

