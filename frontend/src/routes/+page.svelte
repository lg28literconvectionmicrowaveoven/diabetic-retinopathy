<script lang="ts">
	import { onMount } from 'svelte';
	import { analyzeImage, checkHealth, preprocessImage } from '$lib/api';
	import { normalizeRegions } from '$lib/gradcam';
	import { screening } from '$lib/state.svelte';
	import ActionButtons from '$lib/components/ActionButtons.svelte';
	import ImageCanvas from '$lib/components/ImageCanvas.svelte';
	import ImageDropzone from '$lib/components/ImageDropzone.svelte';
	import RawJsonPanel from '$lib/components/RawJsonPanel.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';

	const regions = $derived(normalizeRegions(screening.result));

	const statusText = $derived.by(() => {
		if (screening.error) return screening.error;
		switch (screening.status) {
			case 'idle':
				return 'No image selected.';
			case 'ready':
				return 'Image loaded — press “Preprocess”.';
			case 'preprocessing':
				return 'Preprocessing image…';
			case 'preprocessed':
				return 'Preprocessed — press “Analyze”.';
			case 'analyzing':
				return 'Analyzing preprocessed image…';
			case 'analyzed':
				return 'Analysis complete.';
		}
	});

	onMount(() => {
		checkHealth().then((online) => {
			screening.backendOnline = online;
		});
	});

	async function handlePreprocess() {
		const file = screening.file;
		if (!file || screening.busy) return;
		screening.status = 'preprocessing';
		screening.error = null;
		try {
			const res = await preprocessImage(file);
			if (!res.image) throw new Error('Backend returned no preprocessed image.');
			screening.preprocessedUrl = res.image;
			screening.preprocessingTimeMs =
				typeof res.preprocessing_time_ms === 'number' ? res.preprocessing_time_ms : null;
			screening.status = 'preprocessed';
		} catch (e) {
			screening.error = e instanceof Error ? e.message : String(e);
			screening.status = 'ready';
		}
	}

	async function handleAnalyze() {
		const image = screening.preprocessedUrl;
		if (!image || screening.busy) return;
		screening.status = 'analyzing';
		screening.error = null;
		try {
			screening.result = await analyzeImage(image);
			screening.status = 'analyzed';
		} catch (e) {
			screening.error = e instanceof Error ? e.message : String(e);
			screening.status = 'preprocessed';
		}
	}
</script>

<svelte:head>
	<title>DR Screening — Preprocess &amp; Analyze</title>
</svelte:head>

<header class="header">
	<div>
		<h1>Diabetic Retinopathy Screening</h1>
		<p class="subtitle">Preprocess a fundus image, then analyze the preprocessed result.</p>
	</div>
	<p
		class="backend"
		class:online={screening.backendOnline === true}
		class:offline={screening.backendOnline === false}
	>
		{#if screening.backendOnline === true}
			Backend: online
		{:else if screening.backendOnline === false}
			Backend: unreachable
		{:else}
			Backend: checking…
		{/if}
	</p>
</header>

<main class="layout">
	<section class="panel image-panel">
		<ImageDropzone onselect={(f) => screening.setFile(f)} disabled={screening.busy} />
		<div class="toolbar">
			<ActionButtons
				status={screening.status}
				canPreprocess={screening.canPreprocess}
				canAnalyze={screening.canAnalyze}
				onpreprocess={handlePreprocess}
				onanalyze={handleAnalyze}
			/>
			{#if screening.file}
				<button class="btn ghost" disabled={screening.busy} onclick={() => screening.clear()}>
					Start over
				</button>
			{/if}
			{#if regions.length > 0}
				<label class="overlay-toggle">
					<input type="checkbox" bind:checked={screening.showOverlay} />
					Show Grad-CAM evidence overlay
				</label>
			{/if}
			<span class="status" class:error={screening.error !== null}>{statusText}</span>
		</div>
		<ImageCanvas
			originalUrl={screening.originalUrl}
			preprocessedUrl={screening.preprocessedUrl}
			{regions}
			showOverlay={screening.showOverlay}
		/>
		{#if screening.preprocessingTimeMs !== null}
			<p class="timing">Preprocessing took {screening.preprocessingTimeMs.toFixed(1)} ms</p>
		{/if}
	</section>

	<aside class="side">
		<ResultsPanel result={screening.result} />
		<RawJsonPanel data={screening.result} />
	</aside>
</main>

<style>
	.header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 1rem;
		margin-bottom: 1.25rem;
		flex-wrap: wrap;
	}
	h1 {
		margin: 0;
		font-size: 1.6rem;
	}
	.subtitle {
		margin: 0.25rem 0 0;
		color: #a0aec0;
	}
	.backend {
		margin: 0;
		font-size: 0.85rem;
		font-weight: 600;
		padding: 0.4rem 0.8rem;
		border-radius: 999px;
		background: #2d3748;
		color: #cbd5e0;
		white-space: nowrap;
	}
	.backend.online {
		background: #22543d;
		color: #9ae6b4;
	}
	.backend.offline {
		background: #742a2a;
		color: #feb2b2;
	}
	.layout {
		display: grid;
		grid-template-columns: 3fr 1fr;
		gap: 1rem;
		align-items: start;
	}
	@media (max-width: 1000px) {
		.layout {
			grid-template-columns: 1fr;
		}
	}
	.panel {
		background: #1a202c;
		border: 1px solid #2d3748;
		border-radius: 10px;
		padding: 1.25rem;
	}
	.image-panel {
		display: grid;
		gap: 1rem;
	}
	.toolbar {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.btn.ghost {
		background: transparent;
		border: 1px solid #4a5568;
		color: #cbd5e0;
		border-radius: 8px;
		padding: 0.6rem 1rem;
		font-weight: 600;
		cursor: pointer;
	}
	.btn.ghost:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.overlay-toggle {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		color: #cbd5e0;
		font-size: 0.9rem;
	}
	.status {
		margin-left: auto;
		color: #a0aec0;
		font-size: 0.9rem;
	}
	.status.error {
		color: #fc8181;
	}
	.side {
		display: grid;
		gap: 1rem;
	}
	.timing {
		margin: 0;
		color: #718096;
		font-size: 0.85rem;
	}
</style>
