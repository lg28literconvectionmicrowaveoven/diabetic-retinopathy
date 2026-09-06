-https://pmc.ncbi.nlm.nih.gov/articles/PMC13128382/ (How well do frozen foundation models transfer? A calibration-focused benchmark for diabetic retinopathy grading)

-https://github.com/opisthion06/diabetic_retinopathy/blob/main/DR_Benchmark_v2%20(5).ipynb (relevant minimal implementation)

-https://huggingface.co/google/medsiglip-448 (vision transformer model for encoding general medical images and text)

-https://www.adcis.net/en/third-party/messidor2/ (original dataset) 

-https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades/data (dataset with DR grades third party annotation)

-https://www.mdpi.com/2306-5729/3/3/25 - Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research (validation set)

-https://zenodo.org/records/17219542 (IDRiD dataset download)

-https://ieeexplore.ieee.org/document/11500477 (reference paper)

-https://github.com/justinengelmann/QuickQual (pretrained model for retinal scan quality)

## QuickQual (image quality gate)

Every uploaded scan is quality-checked by QuickQual (DenseNet-121 ImageNet
features + the authors' SVM) before DR grading. Only `good` and `usable`
images proceed; `bad` scans are rejected (`reject_and_reacquire`).

```bash
pip install timm torchvision joblib
```

Download `quickqual_dn121_512.pkl` from the
[QuickQual release](https://github.com/justinengelmann/QuickQual/releases) and
place it at `preproc/quickqual_dn121_512.pkl` (the default path), or point
`QUICKQUAL_MODEL=/path/to/quickqual_dn121_512.pkl` at it.

Notes:
- The gate runs on raw, pre-CLAHE images and never rewrites inputs.
- Without the `.pkl`, the backend falls back to a basic contrast/sharpness
  check and logs a warning — real quality decisions then are not QuickQual's.

## Hugging Face model access (MedSigLIP)

The encoder `google/medsiglip-448` (~3.3 GB) is downloaded from Hugging Face
on first use and cached under `~/.cache/huggingface/hub/`.

1. Create a personal access token at https://huggingface.co/settings/tokens
   (read scope is enough).
2. Authenticate with either:
   ```bash
   huggingface-cli login          # stores the token in ~/.cache/huggingface/token
   ```
   or set an environment variable: `HF_TOKEN=hf_...`
3. Once cached, the backend can run fully offline:
   ```bash
   HF_HUB_OFFLINE=1 python -m uvicorn backend.server:app --port 8000
   ```


##to-do

# 1. point data_root at their MESSIDOR-2 copy + the Kaggle messidor_data.csv
# 2. validate pairing on the real data:
python tests/validate_messidor_pairs.py --config config.yaml --dataset messidor2
# 3. train (produces fold_0..4/best.pt + OOF internal report):
python train.py
# 4. external eval on IDRiD (once its official labels are wired):
python test.py
# 5. send back outputs/checkpoints/multiclass/fold_*/best.pt