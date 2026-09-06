import type {
	AppStatus,
	GradcamRegion,
	NormalizedRect,
	AnalysisResult
} from './types';

function isCornerRegion(region: GradcamRegion): region is Extract<
	GradcamRegion,
	{ x1: number }
> {
	return 'x1' in region && 'y1' in region && 'x2' in region && 'y2' in region;
}

function isPolygonRegion(region: GradcamRegion): region is Extract<
	GradcamRegion,
	{ polygon: Array<[number, number]> }
> {
	return 'polygon' in region && Array.isArray(region.polygon);
}

function clamp01(value: number): number {
	return Math.min(1, Math.max(0, value));
}

/**
 * Normalize the three region forms from the backend contract
 * (rect, corner rect, polygon) — plus the top-level `regions` /
 * `gradcam_regions` fallbacks — into 0..1 rectangles for overlay rendering.
 */
export function normalizeRegions(result: AnalysisResult | null | undefined): NormalizedRect[] {
	if (!result) return [];
	const raw: GradcamRegion[] =
		result.gradcam?.regions ?? result.regions ?? result.gradcam_regions ?? [];
	if (!Array.isArray(raw)) return [];

	const out: NormalizedRect[] = [];
	for (const region of raw) {
		const normalized = region.normalized !== false;
		const score = typeof region.score === 'number' ? region.score : undefined;
		if (isCornerRegion(region)) {
			out.push({
				x: Math.min(region.x1, region.x2),
				y: Math.min(region.y1, region.y2),
				width: Math.abs(region.x2 - region.x1),
				height: Math.abs(region.y2 - region.y1),
				score,
				normalized
			});
		} else if (isPolygonRegion(region)) {
			const xs = region.polygon.map((p) => p[0]);
			const ys = region.polygon.map((p) => p[1]);
			if (xs.length === 0) continue;
			const x0 = Math.min(...xs);
			const y0 = Math.min(...ys);
			out.push({
				x: x0,
				y: y0,
				width: Math.max(...xs) - x0,
				height: Math.max(...ys) - y0,
				score,
				normalized
			});
		} else if (
			typeof region.x === 'number' &&
			typeof region.y === 'number' &&
			typeof region.width === 'number' &&
			typeof region.height === 'number'
		) {
			out.push({ x: region.x, y: region.y, width: region.width, height: region.height, score, normalized });
		}
	}
	return out.map((r) =>
		r.normalized
			? {
					...r,
					x: clamp01(r.x),
					y: clamp01(r.y),
					width: clamp01(r.width),
					height: clamp01(r.height)
				}
			: r
	);
}

export type { AppStatus };
