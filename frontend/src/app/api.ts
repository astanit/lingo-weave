/**
 * Base URL for all API calls to the FastAPI backend.
 * Uses process.env.NEXT_PUBLIC_API_URL (set in build/runtime env, e.g. Railway).
 * Fallback: http://localhost:8000 for local dev.
 */
export function getApiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
}
