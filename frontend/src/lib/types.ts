// Types mirroring the backend JSON contract (see ../gui/BACKEND_JSON_CONTRACT.md).

/** Rectangle region: { x, y, width, height, score?, normalized? } */
export interface GradcamRectRegion {
	x: number;
	y: number;
	width: number;
	height: number;
	score?: number;
	normalized?: boolean;
}

/** Corner rectangle region: { x1, y1, x2, y2, score?, normalized? } */
export interface GradcamCornerRegion {
	x1: number;
	y1: number;
	x2: number;
	y2: number;
	score?: number;
	normalized?: boolean;
}

/** Polygon region: { polygon: [[x, y], ...], score?, normalized? } */
export interface GradcamPolygonRegion {
	polygon: Array<[number, number]>;
	score?: number;
	normalized?: boolean;
}

export type GradcamRegion = GradcamRectRegion | GradcamCornerRegion | GradcamPolygonRegion;

/** Region normalized to a 0..1 rectangle for overlay rendering. */
export interface NormalizedRect {
	x: number;
	y: number;
	width: number;
	height: number;
	score?: number;
	/** False when the backend sent absolute pixel coordinates. */
	normalized: boolean;
}

export interface EvidenceCounts {
	microaneurysms?: number;
	hemorrhages?: number;
	exudates?: number;
	[key: string]: number | undefined;
}

export interface AnalysisResult {
	dr_grade: number;
	confidence: number;
	referable_dr: boolean;
	human_review?: boolean;
	evidence?: EvidenceCounts | null;
	gradcam?: { regions?: GradcamRegion[] } | null;
	/** Accepted as fallbacks, like the Tkinter GUI does. */
	regions?: GradcamRegion[];
	gradcam_regions?: GradcamRegion[];
	explainability_time_ms?: number;
	analyze_time_ms?: number;
	[key: string]: unknown;
}

export interface PreprocessResult {
	/** Preprocessed image as a data URL (e.g. "data:image/png;base64,..."). */
	image: string;
	preprocessing_time_ms?: number;
	width?: number;
	height?: number;
	[key: string]: unknown;
}

export type AppStatus =
	| 'idle'
	| 'ready'
	| 'preprocessing'
	| 'preprocessed'
	| 'analyzing'
	| 'analyzed';
