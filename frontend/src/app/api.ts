/**
 * Base URL for the FastAPI backend. Used only for client-side fetch calls.
 * Set NEXT_PUBLIC_API_URL in the environment (e.g. on Railway) to your backend URL.
 */
export function getApiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
}
