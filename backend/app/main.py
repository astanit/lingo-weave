import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.services.epub_weave import WeaveOptions, weave_epub


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LingoWeave API")

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Very simple in-memory job state (sufficient for MVP / single instance)
JOBS: Dict[str, Dict[str, Optional[str]]] = {}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/upload")
async def upload_epub(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Please upload an .epub file")

    upload_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{upload_id}.epub"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"upload_id": upload_id}


def _process_job(upload_id: str, job_id: str):
    try:
        input_path = str(UPLOADS_DIR / f"{upload_id}.epub")
        if not os.path.exists(input_path):
            JOBS[job_id] = {"status": "error", "error": "Upload not found", "output": None}
            return

        JOBS[job_id] = {"status": "running", "error": None, "output": None}
        _, output_path = weave_epub(
            input_epub_path=input_path,
            outputs_dir=str(OUTPUTS_DIR),
            options=WeaveOptions(bold_translations=True),
        )
        JOBS[job_id] = {"status": "done", "error": None, "output": output_path}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e), "output": None}


@app.post("/api/process")
async def process_epub(background: BackgroundTasks, upload_id: str):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "error": None, "output": None}
    background.add_task(_process_job, upload_id, job_id)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["output"]:
        raise HTTPException(status_code=400, detail="Job not finished")
    path = job["output"]
    return FileResponse(
        path,
        media_type="application/epub+zip",
        filename="lingoweave.epub",
    )


@app.get("/api/config")
async def config():
    # Helps the frontend show a warning if key is missing
    return JSONResponse(
        {
            "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
            "model": "google/gemini-flash-1.5-8b",
        }
    )

