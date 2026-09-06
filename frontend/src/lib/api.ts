import type { AnalysisResult, PreprocessResult } from './types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ||
	'http://127.0.0.1:8000';

async function parseJson(response: Response): Promise<unknown> {
	if (!response.ok) {
		let detail = `${response.status} ${response.statusText}`;
		try {
			const body = (await response.json()) as { detail?: unknown };
			if (typeof body.detail === 'string') detail = body.detail;
		} catch {
			// Fall through to the HTTP status detail.
		}
		throw new Error(`Backend request failed: ${detail}`);
	}
	return response.json();
}

/** GET /health — lightweight backend reachability probe. */
export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
	try {
		const response = await fetch(`${BASE_URL}/health`, { signal });
		if (!response.ok) return false;
		const body = (await response.json()) as { status?: string };
		return body.status === 'pass';
	} catch {
		return false;
	}
}

/**
 * POST /preprocess — upload the raw fundus image, receive the
 * preprocessed image as a data URL. Stateless: the caller holds
 * the returned image and sends it back to `analyzeImage`.
 */
export async function preprocessImage(file: File, signal?: AbortSignal): Promise<PreprocessResult> {
	const form = new FormData();
	form.append('file', file, file.name);
	const response = await fetch(`${BASE_URL}/preprocess`, {
		method: 'POST',
		body: form,
		signal
	});
	return (await parseJson(response)) as PreprocessResult;
}

/**
 * POST /analyze — grade the preprocessed image (data URL previously
 * returned by `preprocessImage`).
 */
export async function analyzeImage(imageDataUrl: string, signal?: AbortSignal): Promise<AnalysisResult> {
	const response = await fetch(`${BASE_URL}/analyze`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ image: imageDataUrl }),
		signal
	});
	return (await parseJson(response)) as AnalysisResult;
}

export { BASE_URL };
