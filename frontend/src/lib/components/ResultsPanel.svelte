<script lang="ts">
	import type { AnalysisResult } from '$lib/types';

	let { result }: { result: AnalysisResult | null } = $props();

	const evidenceEntries = $derived(
		result?.evidence && typeof result.evidence === 'object'
			? Object.entries(result.evidence).filter(([, v]) => typeof v === 'number')
			: []
	);
	const timingMs = $derived(result?.analyze_time_ms ?? result?.explainability_time_ms ?? null);
	const gradeLabel = $derived(
		result === null ? '—' : `Grade ${result.dr_grade} / 4`
	);
</script>

<section class="panel" aria-live="polite">
	<h2>Model Result</h2>
	{#if result === null}
		<p class="empty">No analysis yet. Upload an image, preprocess it, then analyze.</p>
	{:else}
		<p class="label">DR Grade</p>
		<p class="grade">{gradeLabel}</p>
		<dl class="facts">
			<div>
				<dt>Confidence</dt>
				<dd>{(result.confidence * 100).toFixed(1)}%</dd>
			</div>
			<div>
				<dt>Referable DR</dt>
				<dd class:flag={result.referable_dr}>{result.referable_dr ? 'Yes' : 'No'}</dd>
			</div>
			{#if result.human_review !== undefined}
				<div>
					<dt>Human review</dt>
					<dd class:flag={result.human_review}>{result.human_review ? 'Recommended' : 'Not required'}</dd>
				</div>
			{/if}
			{#if timingMs !== null}
				<div>
					<dt>Explainability time</dt>
					<dd>{timingMs.toFixed(1)} ms</dd>
				</div>
			{/if}
		</dl>
		{#if evidenceEntries.length > 0}
			<p class="label">Evidence</p>
			<ul class="evidence">
				{#each evidenceEntries as [name, count] (name)}
					<li><span>{name}</span><strong>{count}</strong></li>
				{/each}
			</ul>
		{/if}
	{/if}
</section>

<style>
	.panel {
		background: #1a202c;
		border: 1px solid #2d3748;
		border-radius: 10px;
		padding: 1.25rem;
	}
	h2 {
		margin: 0 0 0.75rem;
		font-size: 1rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #a0aec0;
	}
	.empty {
		color: #718096;
		margin: 0;
	}
	.label {
		font-size: 0.8rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: #a0aec0;
		margin: 0.75rem 0 0.25rem;
	}
	.grade {
		font-size: 2.25rem;
		font-weight: 800;
		margin: 0;
		color: #fff;
	}
	.facts {
		margin: 0.75rem 0 0;
		display: grid;
		gap: 0.5rem;
	}
	.facts > div {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
	}
	.facts dt {
		color: #a0aec0;
	}
	.facts dd {
		margin: 0;
		font-weight: 600;
		color: #e2e8f0;
	}
	.facts dd.flag {
		color: #fc8181;
	}
	.evidence {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 0.35rem;
	}
	.evidence li {
		display: flex;
		justify-content: space-between;
		background: #2d3748;
		border-radius: 6px;
		padding: 0.4rem 0.7rem;
		color: #e2e8f0;
	}
</style>
