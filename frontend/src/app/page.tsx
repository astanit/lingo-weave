"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE =
  typeof window !== "undefined"
    ? (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
    : "";

type JobStatus = "queued" | "running" | "done" | "error";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<{ has_openrouter_key: boolean } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);

  const fetchConfig = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/config`);
      if (r.ok) setConfig(await r.json());
    } catch {
      setConfig({ has_openrouter_key: false });
    }
  }, []);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const pollJob = useCallback(async (id: string) => {
    const res = await fetch(`${API_BASE}/api/status/${id}`);
    if (!res.ok) return null;
    const data = await res.json();
    setStatus(data.status);
    setError(data.error || null);
    return data;
  }, []);

  useEffect(() => {
    if (!jobId || status === "done" || status === "error") return;
    const t = setInterval(() => pollJob(jobId), 1500);
    return () => clearInterval(t);
  }, [jobId, status, pollJob]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && f.name.toLowerCase().endsWith(".epub")) {
      setFile(f);
      setUploadId(null);
      setJobId(null);
      setStatus(null);
      setError(null);
    } else {
      setFile(null);
    }
  };

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: form,
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Upload failed");
      setUploadId(data.upload_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const process = async () => {
    if (!uploadId) return;
    setProcessing(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/process?upload_id=${uploadId}`, {
        method: "POST",
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Process failed");
      setJobId(data.job_id);
      setStatus("queued");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Process failed");
    } finally {
      setProcessing(false);
    }
  };

  const download = () => {
    if (!jobId) return;
    window.open(`${API_BASE}/api/download/${jobId}`, "_blank");
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col items-center justify-center p-6">
      <main className="w-full max-w-lg space-y-8">
        <header className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-zinc-50">
            LingoWeave
          </h1>
          <p className="mt-2 text-zinc-400">
            Russian EPUB → Diglot Weave (0% → 100% English, bolded)
          </p>
        </header>

        {config && !config.has_openrouter_key && (
          <div
            className="rounded-lg border border-amber-500/50 bg-amber-500/10 px-4 py-3 text-amber-200 text-sm"
            role="alert"
          >
            OpenRouter API key is not set on the server. Set{" "}
            <code className="font-mono">OPENROUTER_API_KEY</code> in your
            environment (e.g. Railway Variables) for translation to work.
          </div>
        )}

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 space-y-4">
          <label className="block text-sm font-medium text-zinc-300">
            ­Choose EPUB
          </label>
          <input
            type="file"
            accept=".epub"
            onChange={onFileChange}
            className="block w-full text-sm text-zinc-400 file:mr-4 file:rounded-lg file:border-0 file:bg-zinc-700 file:px-4 file:py-2 file:text-zinc-100 file:cursor-pointer hover:file:bg-zinc-600"
          />
          {file && (
            <p className="text-sm text-zinc-500">
              {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </p>
          )}

          <div className="flex flex-wrap gap-3 pt-2">
            <button
              type="button"
              onClick={upload}
              disabled={!file || uploading}
              className="rounded-lg bg-zinc-700 px-4 py-2 text-sm font-medium text-zinc-100 hover:bg-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? "Uploading…" : "Upload"}
            </button>
            <button
              type="button"
              onClick={process}
              disabled={!uploadId || processing}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {processing ? "Starting…" : "Process (Weave)"}
            </button>
            {status === "done" && jobId && (
              <button
                type="button"
                onClick={download}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
              >
                Download EPUB
              </button>
            )}
          </div>
        </div>

        {(status || error) && (
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            {status && (
              <p className="text-sm text-zinc-300">
                Status:{" "}
                <span
                  className={
                    status === "done"
                      ? "text-emerald-400"
                      : status === "error"
                        ? "text-red-400"
                        : "text-amber-400"
                  }
                >
                  {status}
                </span>
              </p>
            )}
            {error && (
              <p className="mt-2 text-sm text-red-400" role="alert">
                {error}
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
