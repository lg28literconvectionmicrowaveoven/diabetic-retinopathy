<script lang="ts">
	import type { AppStatus } from '$lib/types';

	let {
		status,
		canPreprocess,
		canAnalyze,
		onpreprocess,
		onanalyze
	}: {
		status: AppStatus;
		canPreprocess: boolean;
		canAnalyze: boolean;
		onpreprocess: () => void;
		onanalyze: () => void;
	} = $props();

	const preprocessing = $derived(status === 'preprocessing');
	const analyzing = $derived(status === 'analyzing');
</script>

<div class="actions">
	<button
		class="btn primary"
		disabled={!canPreprocess || preprocessing || analyzing}
		onclick={onpreprocess}
		title={canPreprocess ? 'Send the uploaded image to POST /preprocess' : 'Upload an image first'}
	>
		{preprocessing ? 'Preprocessing…' : '1. Preprocess'}
	</button>
	<button
		class="btn accent"
		disabled={!canAnalyze || preprocessing || analyzing}
		onclick={onanalyze}
		title={canAnalyze
			? 'Send the preprocessed image to POST /analyze'
			: 'Preprocess an image first'}
	>
		{analyzing ? 'Analyzing…' : '2. Analyze'}
	</button>
</div>

<style>
	.actions {
		display: flex;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.btn {
		font-size: 1rem;
		font-weight: 700;
		padding: 0.7rem 1.4rem;
		border: none;
		border-radius: 8px;
		cursor: pointer;
		color: #fff;
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.btn.primary {
		background: #2b6cb0;
	}
	.btn.primary:hover:not(:disabled) {
		background: #2c5282;
	}
	.btn.accent {
		background: #276749;
	}
	.btn.accent:hover:not(:disabled) {
		background: #22543d;
	}
</style>
