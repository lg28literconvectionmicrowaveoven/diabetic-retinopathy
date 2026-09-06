import type { AnalysisResult, AppStatus } from './types';

/**
 * Shared screening flow state (Svelte 5 runes).
 *
 * Flow: idle → ready → preprocessing → preprocessed → analyzing → analyzed.
 * A newly selected file resets everything downstream of the upload.
 */
class ScreeningState {
	file = $state<File | null>(null);
	originalUrl = $state<string | null>(null);
	preprocessedUrl = $state<string | null>(null);
	preprocessingTimeMs = $state<number | null>(null);
	status = $state<AppStatus>('idle');
	result = $state<AnalysisResult | null>(null);
	error = $state<string | null>(null);
	showOverlay = $state(true);
	backendOnline = $state<boolean | null>(null);

	readonly canPreprocess = $derived(
		(this.status === 'ready' || this.status === 'preprocessed' || this.status === 'analyzed') &&
			this.file !== null
	);
	readonly canAnalyze = $derived(
		(this.status === 'preprocessed' || this.status === 'analyzed') &&
			this.preprocessedUrl !== null
	);
	readonly busy = $derived(this.status === 'preprocessing' || this.status === 'analyzing');

	setFile(file: File) {
		if (this.originalUrl?.startsWith('blob:')) URL.revokeObjectURL(this.originalUrl);
		this.file = file;
		this.originalUrl = URL.createObjectURL(file);
		this.preprocessedUrl = null;
		this.preprocessingTimeMs = null;
		this.result = null;
		this.error = null;
		this.status = 'ready';
	}

	clear() {
		if (this.originalUrl?.startsWith('blob:')) URL.revokeObjectURL(this.originalUrl);
		this.file = null;
		this.originalUrl = null;
		this.preprocessedUrl = null;
		this.preprocessingTimeMs = null;
		this.result = null;
		this.error = null;
		this.status = 'idle';
	}
}

export const screening = new ScreeningState();
