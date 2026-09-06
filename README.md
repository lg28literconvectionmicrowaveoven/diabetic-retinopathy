# Inmarscan

> Diabetic retinopathy tele-screening for rural Indian healthcare centres using Google MedSigLIP, fast Grad-CAM explainability, and district-level discrete-event queuing simulation.

---

## Why we built this

India has over **101 million people living with diabetes**, and more than 70% of them live in rural areas. Diabetic Retinopathy (DR) causes progressive blindness, but in its early stages (Grades 1-2) patients feel zero symptoms.

The bottleneck isn't cameras—thousands of rural Primary Healthcare Centres (PHCs) have basic non-mydriatic fundus cameras. The bottleneck is **specialists**:
- Rural PHCs have nurses and community health officers, but **zero ophthalmologists**.
- Patients who need treatment are told to travel 60-100 km to a district or tier-1 hospital, so most simply don't go until their vision is permanently damaged.
- Bandwidth at rural PHCs is often slow (256 kbps - 2 Mbps), and field scans often have cataract fog or low contrast.

**Inmarscan** is an end-to-end triage and screening pipeline designed specifically for this setup: it screens retinal scans locally, filters out clearly healthy eyes, flags referable disease (Grade 2+), generates visual heatmaps so a remote eye doctor can verify the scan in under 30 seconds, and includes a full Simulink simulation proving this can scale to 100,000+ patients a year in a real district.

---

## How it works

```text
[Rural PHC]
  1. Nurse captures fundus photo on portable camera (~3 min)
  2. Edge quality gate checks if the image is readable
     -> If blurry / cataract-blinded, immediate retake on the spot (<1% rate)
     -> If low-contrast, adaptive green-channel CLAHE enhancement
  3. Image compressed to ~428 KB and queued over rural uplink (1.7s at 2 Mbps)
          │
          ▼
[Central Server / GPU Hub]
  4. Google MedSigLIP ViT (frozen SO400M) extracts 1152-dim embeddings
  5. 5-fold ensemble MLP heads predict DR grade (0 to 4) & referral
  6. Analytical VJP Grad-CAM generates lesion heatmaps (~0.4s)
          │
          ├─► Grade 0-1 (No DR / Mild): Auto-cleared at PHC (38% of patients, 0 MD time)
          │
          └─► Grade 2-4 (Referable DR): Queued to Remote Ophthalmologist
                  │
                  ▼
[Remote Doctor Tele-Review]
  7. Doctor reviews scan with Grad-CAM lesion heatmap overlay in < 30 seconds
  8. Confirmed cases get routed to district hospital for laser/vitrectomy
```

---

## Key Numbers & Results

We trained our classification heads on **Messidor-2** using **Grouped 5-Fold Cross-Validation** (both eyes of every patient strictly kept in the same fold to prevent data leakage) and validated externally on the **official Indian IDRiD test cohort** (66 patients, 103 eyes).

### Diagnostic Accuracy

| Task | Metric | Indian IDRiD (Patient-Level) | Messidor-2 (Patient-Level) | Target |
| :--- | :--- | :---: | :---: | :---: |
| **Referral Triage** | **Sensitivity (Recall)** | **100.0% (45/45)** | 76.05% (200/263) | $> 90\%$ (Zero missed cases) |
| **Referral Triage** | Accuracy | **81.82% (54/66)** | 88.08% (783/889) | $> 80\%$ |
| **Referral Triage** | **ROC-AUC** | **0.9312** | **0.9469** | $> 0.85$ |
| **Referral Triage** | Specificity | 52.38% (11/21) | 97.07% (607/626) | $> 50\%$ |
| **Referral Triage** | Negative Predictive Value | **100.0%** | 90.60% | Zero missed referrals |
| **5-Grade Staging** | Quadratic Weighted Kappa (QWK) | 0.4248 | **0.7901** | High ordinal agreement |
| **5-Grade Staging** | Exact Multiclass Accuracy | 33.33% | 71.54% | Exact grade match |

> **Bottom Line:** On the real Indian clinical cohort (IDRiD), our model caught **45 out of 45 referable patients (100% sensitivity)**. No sight-threatening DR was missed.

---

## District Scaling Simulation (Simulink)

We didn't just train a classifier—we built a full discrete-event queuing simulation in MATLAB/Simulink (`DR_Rural_India_Simulink/`) to answer the real health-system question:

> *What does it actually take to screen 100,000+ diabetic patients across a rural district every year?*

We benchmarked every step on real hardware (Apple M4 GPU, 1.17s inference, 0.41s Grad-CAM, 428 KB payload, 2 Mbps rural network) and simulated 6 district setups:

| Scenario | PHCs | Cameras/PHC | Bandwidth | AI Workers | Eye Doctors | Annual Capacity | Bottleneck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Single PHC Pilot | 5 | 1 | 1.0 Mbps | 1 GPU | 1 MD | 15,000 pts/yr | Camera |
| Typical Sub-District | 15 | 1 | 2.0 Mbps | 2 GPUs | 2 MDs | 56,250 pts/yr | Eye Doctors |
| Bandwidth-Constrained (2G) | 25 | 1 | 256 kbps | 2 GPUs | 2 MDs | 93,750 pts/yr | Eye Doctors |
| Specialist-Constrained | 35 | 1 | 2.0 Mbps | 4 GPUs | 1 MD | 131,250 pts/yr | Eye Doctors |
| **100k District (Balanced)** | **30** | **1** | **2.0 Mbps** | **4 GPUs** | **3 MDs** | **120,000 pts/yr** | **Eye Doctors** |
| High-Density District | 50 | 2 | 4.0 Mbps | 6 GPUs | 5 MDs | 250,000 pts/yr | Eye Doctors |

### What this proves:
1. **100k goal is realistic:** 30 rural PHCs with 1 camera each, backed by 4 GPU workers and 3 remote eye doctors, can screen **120,000 patients a year**.
2. **AI removes the doctor bottleneck:** By auto-clearing 38% of negative scans and giving doctors Grad-CAM heatmaps to inspect the rest in under 30 seconds, 3 doctors can handle an entire district.

---

## Quickstart

Clone and run using the helper script:

```bash
git clone https://github.com/Shiviatrix/diabetic-retinopathy.git
cd diabetic-retinopathy

# Launch full web app (FastAPI backend + SvelteKit frontend)
./launch.sh --all

# Or launch the desktop GUI (for offline PHC clinic laptops)
./launch.sh --gui

# Run external evaluation on IDRiD
./launch.sh --eval

# Run the test suite (18 unit/integration tests)
./launch.sh --test
```

### Manual Run

```bash
# Backend (FastAPI on :8000)
python3 -m uvicorn backend.server:app --port 8000

# Frontend (SvelteKit on :5173)
cd frontend && npm install && npm run dev -- --port 5173

# Desktop GUI (Tkinter)
python3 gui/gui.py
```

---

## What's in the repo

```text
├── backend/
│   └── server.py                 # FastAPI server (/preprocess, /analyze, /predict, /health)
├── frontend/                     # SvelteKit web interface with Grad-CAM heatmap slider
├── gui/
│   ├── gui.py                    # Standalone Tkinter desktop app for rural nurse stations
│   └── samples/                  # 10 clinical fundus sample images (Grades 0-4)
├── models/
│   └── medsiglip.py              # Frozen MedSigLIP ViT encoder + MLP classification head
├── tests/                        # 18 pytest unit and integration tests
├── results/
│   ├── presentation_assets/      # 300-DPI Black & White tables & architecture charts
│   ├── simulink_district_capacity_matrix.csv
│   └── simulink_parameters.json  # Empirical hardware benchmarks
├── DR_Rural_India_Simulink/      # Full MATLAB/Simulink discrete-event simulation model
│   ├── DR_Rural_India.slx        # Simulink model file
│   ├── DR_simulink_parameters.m  # Model parameters
│   └── run_simulation.m          # Simulation run script
├── launch.sh                     # Master launch script
├── train.py                      # Grouped 5-fold training pipeline
├── test.py                       # External evaluation runner
├── pair_eyes_patient_level.py    # Bilateral eye pairing & patient-level metrics
├── explainability.py             # Vector-Jacobian Product (VJP) Grad-CAM engine
├── config.yaml                   # Dataset paths and grouping config
└── best.pt                       # Primary trained model checkpoint
```

---

## Datasets & References

- **Indian Diabetic Retinopathy Image Dataset (IDRiD):** Prasanna et al., *MDPI Data* (2018). [doi:10.3390/data3030025](https://www.mdpi.com/2306-5729/3/3/25).
- **Messidor-2:** Decencière et al., *ADCIS* (2014) with Kaggle DR third-party annotations.
- **MedSigLIP:** Google Research medical vision transformer ([google/medsiglip-448](https://huggingface.co/google/medsiglip-448)).
- **QuickQual:** Justin Engelmann, retinal scan quality gate ([GitHub](https://github.com/justinengelmann/QuickQual)).
- **ICMR-INDIAB Study:** The Lancet Diabetes & Endocrinology (2023).