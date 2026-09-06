# DR Screening GUI

Tkinter GUI for the MedSigLIP + MLP diabetic-retinopathy backend.

## Install

```bash
pip install -r requirements.txt
```

For drag-and-drop support:

```bash
pip install tkinterdnd2
```

If `tkinterdnd2` is not installed, Upload Image still works.

On Linux, the system Tk package may also be required (`python3-tk`).

## Run

```bash
python gui.py
```

Put demonstration images in:

```text
samples/
```

and they will appear in the Sample Images list.

Set the backend address in `gui_config.yaml`.

See `BACKEND_JSON_CONTRACT.md` for the expected JSON format.
