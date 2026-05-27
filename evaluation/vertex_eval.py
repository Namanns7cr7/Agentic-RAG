import json
import os
from pathlib import Path


def _load_cases() -> dict:
    results_path = Path(__file__).with_name("results.json")
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    return json.loads(Path(__file__).with_name("eval_queries.json").read_text(encoding="utf-8"))


def main() -> None:
    data = _load_cases()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        print("Vertex evaluation skipped: GOOGLE_CLOUD_PROJECT is not set.")
        return

    try:
        import vertexai
        from vertexai.preview.evaluation import EvalTask
    except Exception as exc:
        print(f"Vertex evaluation skipped: {exc}")
        return

    vertexai.init(project=project, location=location)

    # This is a lightweight placeholder that validates availability of the Eval API.
    # It does not run costly evaluation jobs by default.
    try:
        task = EvalTask()
        results = {
            "status": "available",
            "project": project,
            "location": location,
            "cases_loaded": len(data.get("cases", [])) if isinstance(data, dict) else 0,
        }
        output_path = Path(__file__).with_name("vertex_results.json")
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Vertex evaluation stub completed. Results saved to evaluation/vertex_results.json")
    except Exception as exc:
        print(f"Vertex evaluation failed: {exc}")


if __name__ == "__main__":
    main()
