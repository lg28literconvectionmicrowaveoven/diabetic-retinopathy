<script lang="ts">
	let {
		data,
		title = 'Backend JSON'
	}: {
		data: unknown;
		title?: string;
	} = $props();

	const pretty = $derived(JSON.stringify(data, null, 2));
</script>

<section class="panel">
	<h2>{title}</h2>
	{#if data === null || data === undefined}
		<p class="empty">Nothing received yet.</p>
	{:else}
		<pre><code>{pretty}</code></pre>
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
	pre {
		margin: 0;
		max-height: 320px;
		overflow: auto;
		background: #171923;
		border-radius: 6px;
		padding: 0.75rem;
	}
	code {
		font-family: Consolas, Menlo, monospace;
		font-size: 0.8rem;
		color: #cbd5e0;
	}
</style>
