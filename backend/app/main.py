import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.services.epub_weave import WeaveOptions, weave_epub
from app.services.txt_weave import weave_txt
from app.services.fb2_weave import weave_fb2


BASE_DIR = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = (".epub", ".txt", ".fb2")
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        from telegram_bot import run_telegram_bot
        task = asyncio.create_task(run_telegram_bot())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    else:
        yield


app = FastAPI(title="LingoWeave API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить всем
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Very simple in-memory job state (sufficient for MVP / single instance)
JOBS: Dict[str, Dict[str, Optional[str]]] = {}


@app.get("/health")
def health():
    return {"ok": True}


def _get_upload_extension(filename: str) -> str:
    if not filename:
        return ""
    lower = filename.lower()
    for ext in ALLOWED_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return ""


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = _get_upload_extension(file.filename or "")
    if not ext:
        raise HTTPException(
            status_code=400,
            detail="Please upload an .epub, .txt, or .fb2 file",
        )

    upload_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{upload_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"upload_id": upload_id, "extension": ext}


def _process_job(upload_id: str, job_id: str):
    try:
        input_path = None
        ext = None
        for e in ALLOWED_EXTENSIONS:
            candidate = UPLOADS_DIR / f"{upload_id}{e}"
            if candidate.exists():
                input_path = str(candidate)
                ext = e
                break
        if not input_path or not os.path.exists(input_path):
            JOBS[job_id] = {"status": "error", "error": "Upload not found", "output": None}
            return

        JOBS[job_id] = {"status": "running", "error": None, "output": None}
        opts = WeaveOptions(bold_translations=True)

        if ext == ".epub":
            _, output_path = weave_epub(
                input_epub_path=input_path,
                outputs_dir=str(OUTPUTS_DIR),
                options=opts,
            )
        elif ext == ".txt":
            _, output_path = weave_txt(
                input_txt_path=input_path,
                outputs_dir=str(OUTPUTS_DIR),
                options=opts,
            )
        elif ext == ".fb2":
            _, output_path = weave_fb2(
                input_fb2_path=input_path,
                outputs_dir=str(OUTPUTS_DIR),
                options=opts,
            )
        else:
            JOBS[job_id] = {"status": "error", "error": "Unsupported format", "output": None}
            return

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


def _media_type_and_filename(path: str) -> tuple:
    p = path.lower()
    if p.endswith(".txt"):
        return "text/plain; charset=utf-8", "lingoweave.txt"
    if p.endswith(".fb2"):
        return "application/xml", "lingoweave.fb2"
    return "application/epub+zip", "lingoweave.epub"


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["output"]:
        raise HTTPException(status_code=400, detail="Job not finished")
    path = job["output"]
    media_type, filename = _media_type_and_filename(path)
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/config")
async def config():
    # API key and model from env (see app.services.openrouter_translate).
    return JSONResponse(
        {
            "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
            "model": os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5"),
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

