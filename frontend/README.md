# DR Screening Frontend (SvelteKit)

Web UI for diabetic-retinopathy screening with a two-step flow:

1. **Preprocess** — uploads the raw fundus image to `POST /preprocess`.
2. **Analyze** — sends the returned preprocessed image to `POST /analyze` and
   renders the DR grade, confidence, evidence counts, and Grad-CAM overlay.

The app is a static SPA: the browser talks to FastAPI directly
(no SvelteKit server routes involved).

## Run

```bash
npm install
npm run dev
```

Point it at the backend with `VITE_API_BASE_URL` (see `.env.example`,
default `http://127.0.0.1:8000`).

## Expected backend contract

- `POST /preprocess` (multipart `file`) →
  `{ image: "data:image/png;base64,...", preprocessing_time_ms?, width?, height? }`
- `POST /analyze` (JSON `{ image: "<data url>" }`) →
  `{ dr_grade, confidence, referable_dr, human_review?, evidence?,
  gradcam: { regions: [...] }, analyze_time_ms? }`

Region shapes follow `../gui/BACKEND_JSON_CONTRACT.md`
(rect, corner rect, polygon; normalized coords preferred).

## Build

```bash
npm run check
npm run build   # static output in build/
```
