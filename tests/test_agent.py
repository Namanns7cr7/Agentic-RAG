from agentic_rag.rag.config import Settings
from agentic_rag.rag.pipeline import Pipeline

def test_calculator_tool():
    pipe = Pipeline(Settings(), seed_docs=[])
    out = pipe.answer("calculate 2*(3+4)")
    assert "14" in out["final"] or "14" in out["draft"]
