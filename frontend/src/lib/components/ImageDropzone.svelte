<script lang="ts">
	let {
		onselect,
		disabled = false
	}: {
		onselect: (file: File) => void;
		disabled?: boolean;
	} = $props();

	let input: HTMLInputElement | null = $state(null);
	let dragging = $state(false);
	let dragDepth = 0;

	function pickFile() {
		if (!disabled) input?.click();
	}

	function handleFiles(files: FileList | null | undefined) {
		if (disabled || !files || files.length === 0) return;
		const image = [...files].find((f) => f.type.startsWith('image/')) ?? files[0];
		onselect(image);
	}
</script>

<div
	class="dropzone"
	class:dragging
	class:disabled
	role="button"
	tabindex={disabled ? -1 : 0}
	aria-label="Upload fundus image"
	onclick={pickFile}
	onkeydown={(e) => {
		if (e.key === 'Enter' || e.key === ' ') pickFile();
	}}
	ondragenter={(e) => {
		e.preventDefault();
		if (disabled) return;
		dragDepth += 1;
		dragging = true;
	}}
	ondragover={(e) => e.preventDefault()}
	ondragleave={(e) => {
		e.preventDefault();
		dragDepth = Math.max(0, dragDepth - 1);
		if (dragDepth === 0) dragging = false;
	}}
	ondrop={(e) => {
		e.preventDefault();
		dragDepth = 0;
		dragging = false;
		handleFiles(e.dataTransfer?.files);
	}}
>
	<input
		bind:this={input}
		type="file"
		accept="image/*"
		hidden
		{disabled}
		onchange={(e) => {
			handleFiles(e.currentTarget.files);
			e.currentTarget.value = '';
		}}
	/>
	{#if dragging}
		<span class="hint">Drop the image to load it</span>
	{:else}
		<span class="hint"><strong>Upload image</strong> or drag &amp; drop a fundus photo here</span>
	{/if}
</div>

<style>
	.dropzone {
		border: 2px dashed #4a5568;
		border-radius: 10px;
		padding: 1rem;
		text-align: center;
		cursor: pointer;
		background: #1a202c;
		color: #cbd5e0;
		user-select: none;
	}
	.dropzone:hover:not(.disabled),
	.dropzone:focus-visible:not(.disabled) {
		border-color: #63b3ed;
		outline: none;
	}
	.dropzone.dragging {
		border-color: #63b3ed;
		background: #2a4365;
		color: #fff;
	}
	.dropzone.disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.hint {
		font-size: 0.95rem;
	}
	.hint strong {
		color: #fff;
	}
</style>
