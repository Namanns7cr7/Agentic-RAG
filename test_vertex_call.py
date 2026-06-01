import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.getenv("VERTEX_MODEL", "gemini-2.0-flash")

if not PROJECT_ID:
    import subprocess
    try:
        res = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True, check=True)
        gcloud_project = res.stdout.strip()
        if gcloud_project and gcloud_project != "(unset)":
            PROJECT_ID = gcloud_project
            print(f"Fallback to active gcloud config project: {PROJECT_ID}")
    except Exception:
        pass

if not PROJECT_ID or PROJECT_ID == "your-real-project-id":
    raise ValueError(
        "Please specify a valid GOOGLE_CLOUD_PROJECT. You can either:\n"
        "1. Set GOOGLE_CLOUD_PROJECT=your-actual-gcp-project-id in your .env file\n"
        "2. Set it on your CLI using: gcloud config set project your-actual-gcp-project-id"
    )

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

response = client.models.generate_content(
    model=MODEL,
    contents="Say hello from Vertex AI in one short sentence.",
    config=GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=50,
    ),
)

print("Vertex AI response:")
print(response.text)