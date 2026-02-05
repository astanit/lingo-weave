# LingoWeave

Web app that converts Russian EPUB books into a **Diglot Weave** format by progressively increasing the percentage of English words from 0% to 100% chapter-by-chapter. English words are **bolded**.

## Architecture

- **Backend**: FastAPI (`backend/`) — EPUB upload, chapter processing, Diglot Weave generation, download output EPUB.
- **Frontend**: Next.js + Tailwind (`frontend/`) — dark-mode UI to upload, configure, and download results.
- **AI**: OpenRouter via environment variable `OPENROUTER_API_KEY` (never hardcoded).

## Local dev

### Backend

1. Create `backend/.env` (optional) and set:

   - `OPENROUTER_API_KEY=...`

2. Install + run:

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend expects backend at `http://localhost:8000` by default.

## Deployment (Railway)

1. Create a new project and connect your GitHub repo (`lingo-weave`).
2. Railway can deploy from the repo root with `docker-compose.yml`, or you can add two services (backend + frontend) and set each to use its `Dockerfile`.
3. **Variables (critical):** In the Railway project → **Variables** tab, add:
   - `OPENROUTER_API_KEY` = your key (e.g. `sk-or-v1-...`).
   - After the first deploy, set `NEXT_PUBLIC_API_URL` to your **backend** service public URL (e.g. `https://your-backend.up.railway.app`), then redeploy the frontend so the browser can call the API.
   - Set `CORS_ORIGINS` to your **frontend** public URL (e.g. `https://your-frontend.up.railway.app`) so the backend allows requests from the frontend.
4. Redeploy frontend after setting `NEXT_PUBLIC_API_URL` so the build picks it up.

