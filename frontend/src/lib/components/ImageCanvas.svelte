<script lang="ts">
	import type { NormalizedRect } from '$lib/types';

	let {
		originalUrl,
		preprocessedUrl,
		regions = [],
		showOverlay = true
	}: {
		originalUrl: string | null;
		preprocessedUrl: string | null;
		regions?: NormalizedRect[];
		showOverlay?: boolean;
	} = $props();

	// Natural size of the preprocessed image, used to scale regions
	// that arrive in absolute pixel coordinates (normalized === false).
	let naturalWidth = $state(0);
	let naturalHeight = $state(0);

	interface FracRect {
		x: number;
		y: number;
		width: number;
		height: number;
		score?: number;
	}

	const displayRegions = $derived<FracRect[]>(
		showOverlay
			? regions.map((r) => {
					if (r.normalized || naturalWidth <= 0 || naturalHeight <= 0) return r;
					return {
						x: r.x / naturalWidth,
						y: r.y / naturalHeight,
						width: r.width / naturalWidth,
						height: r.height / naturalHeight,
						score: r.score
					};
				})
			: []
	);
</script>

<div class="images">
	<figure class="frame">
		<figcaption>Original</figcaption>
		<div class="canvas">
			{#if originalUrl}
				<img src={originalUrl} alt="Original fundus upload" />
			{:else}
				<span class="placeholder">No image selected</span>
			{/if}
		</div>
	</figure>

	<figure class="frame">
		<figcaption>Preprocessed</figcaption>
		<div class="canvas">
			{#if preprocessedUrl}
				<img
					src={preprocessedUrl}
					alt="Preprocessed fundus"
					onload={(e) => {
						const img = e.currentTarget as HTMLImageElement;
						naturalWidth = img.naturalWidth;
						naturalHeight = img.naturalHeight;
					}}
				/>
				{#if displayRegions.length > 0}
					<svg class="overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
						{#each displayRegions as region, i (i)}
							<rect
								x={region.x}
								y={region.y}
								width={region.width}
								height={region.height}
								vector-effect="non-scaling-stroke"
							>
								<title>
									Evidence region {i + 1}{region.score !== undefined
										? ` (score ${region.score.toFixed(2)})`
										: ''}
								</title>
							</rect>
						{/each}
					</svg>
				{/if}
			{:else}
				<span class="placeholder">Press “Preprocess” to see the model input</span>
			{/if}
		</div>
	</figure>
</div>

<style>
	.images {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
	}
	@media (max-width: 700px) {
		.images {
			grid-template-columns: 1fr;
		}
	}
	.frame {
		margin: 0;
	}
	.frame figcaption {
		font-size: 0.8rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: #a0aec0;
		margin-bottom: 0.4rem;
	}
	.canvas {
		position: relative;
		background: #151515;
		border-radius: 8px;
		min-height: 240px;
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
	}
	.canvas img {
		display: block;
		width: 100%;
		height: auto;
		max-height: 480px;
		object-fit: contain;
	}
	.overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}
	.overlay rect {
		fill: rgba(255, 80, 80, 0.18);
		stroke: #ff5050;
		stroke-width: 2px;
	}
	.placeholder {
		color: #718096;
		font-size: 0.9rem;
		padding: 2rem 1rem;
		text-align: center;
	}
</style>
