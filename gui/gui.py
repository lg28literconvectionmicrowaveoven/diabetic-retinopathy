from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import requests
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml
from PIL import Image, ImageDraw, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def load_config() -> dict[str, Any]:
    with (Path(__file__).with_name("gui_config.yaml")).open(
        "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


class DRScreeningGUI:
    def __init__(self, root: tk.Tk, config: dict[str, Any]):
        self.root = root
        self.cfg = config

        self.backend_url = self.cfg["backend"]["url"].rstrip("/")
        self.endpoint = self.cfg["backend"].get("endpoint", "/predict")
        self.timeout = int(self.cfg["backend"].get("timeout_seconds", 120))

        self.project_root = Path(__file__).resolve().parent
        self.sample_dir = self.project_root / self.cfg["samples"]["directory"]

        self.image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.photo = None
        self.overlay_enabled = tk.BooleanVar(value=True)
        self._regions: list[dict[str, Any]] = []
        self._sample_paths: list[Path] = []

        self._build_style()
        self._build_ui()
        self._load_samples()

        if DND_AVAILABLE:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop_hint.configure(
                text="Drag & drop requires tkinterdnd2. Use Upload Image."
            )

    def _build_style(self):
        self.root.title(
            self.cfg["ui"].get("title", "DR Screening - MedSigLIP")
        )
        self.root.geometry(
            self.cfg["ui"].get("geometry", "1280x800")
        )
        self.root.minsize(1050, 700)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel", font=("Segoe UI", 20, "bold")
        )
        style.configure(
            "Subtitle.TLabel", font=("Segoe UI", 10)
        )
        style.configure(
            "Grade.TLabel", font=("Segoe UI", 32, "bold")
        )
        style.configure(
            "Section.TLabel", font=("Segoe UI", 12, "bold")
        )
        style.configure(
            "Result.TLabel", font=("Segoe UI", 11)
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=8,
        )

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(
            header,
            text="Diabetic Retinopathy Screening",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="MedSigLIP + MLP Classifier Head backend with Grad-CAM evidence overlay",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        main = ttk.Frame(outer)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # image
        image_panel = ttk.LabelFrame(
            main, text="Drag-drop / upload image for screening or select an image from given samples", padding=12
        )
        image_panel.grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )
        image_panel.rowconfigure(1, weight=1)
        image_panel.columnconfigure(0, weight=1)

        controls = ttk.Frame(image_panel)
        controls.grid(
            row=0, column=0, sticky="ew", pady=(0, 10)
        )

        ttk.Button(
            controls,
            text="Upload Image",
            command=self._choose_image,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Analyze",
            style="Action.TButton",
            command=self._analyze,
        ).pack(side="left", padx=8)

        ttk.Checkbutton(
            controls,
            text="Show Grad-CAM evidence overlay",
            variable=self.overlay_enabled,
            command=self._refresh_image,
        ).pack(side="left", padx=8)

        self.status_var = tk.StringVar(
            value="No image selected."
        )
        ttk.Label(
            controls,
            textvariable=self.status_var,
        ).pack(side="right")

        canvas_frame = ttk.Frame(image_panel)
        canvas_frame.grid(
            row=1, column=0, sticky="nsew"
        )
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#151515",
            highlightthickness=0,
        )
        self.canvas.grid(
            row=0, column=0, sticky="nsew"
        )

        self.drop_hint = ttk.Label(
            canvas_frame,
            background="#151515",
            foreground="#FFFFFF",
            text="Drag-drop / upload image for screening\n or select an image from given samples",
            anchor="center",
            font=("Segoe UI", 14),
        )
        self.drop_hint.place(
            relx=0.5, rely=0.5, anchor="center"
        )

        # controls n result
        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)
        side.rowconfigure(3, weight=1)

        sample_box = ttk.LabelFrame(
            side, text="Sample Images", padding=10
        )
        sample_box.grid(
            row=0, column=0, sticky="ew", pady=(0, 10)
        )
        sample_box.columnconfigure(0, weight=1)

        self.sample_list = tk.Listbox(
            sample_box,
            height=8,
            exportselection=False,
            font=("Segoe UI", 10),
        )
        self.sample_list.grid(
            row=0, column=0, sticky="ew"
        )
        self.sample_list.bind(
            "<Double-Button-1>", self._select_sample
        )

        ttk.Button(
            sample_box,
            text="Use Selected Sample",
            command=self._select_sample,
        ).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )

        result_box = ttk.LabelFrame(
            side, text="Model Result", padding=12
        )
        result_box.grid(
            row=1, column=0, sticky="nsew", pady=(0, 10)
        )
        result_box.columnconfigure(0, weight=1)

        ttk.Label(
            result_box,
            text="DR Grade",
            style="Section.TLabel",
        ).pack(anchor="w")

        self.grade_var = tk.StringVar(value="—")
        ttk.Label(
            result_box,
            textvariable=self.grade_var,
            style="Grade.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        self.confidence_var = tk.StringVar(
            value="Confidence: —"
        )
        ttk.Label(
            result_box,
            textvariable=self.confidence_var,
            style="Result.TLabel",
        ).pack(anchor="w")

        self.referable_var = tk.StringVar(
            value="Referable DR: —"
        )
        ttk.Label(
            result_box,
            textvariable=self.referable_var,
            style="Result.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.review_var = tk.StringVar(
            value="Human review: —"
        )
        ttk.Label(
            result_box,
            textvariable=self.review_var,
            style="Result.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.explain_time_var = tk.StringVar(
            value="Explainability: —"
        )
        ttk.Label(
            result_box,
            textvariable=self.explain_time_var,
            style="Result.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        evidence_box = ttk.LabelFrame(
            side, text="Lesion / Evidence", padding=10
        )
        evidence_box.grid(
            row=2, column=0, sticky="nsew", pady=(0, 10)
        )
        evidence_box.rowconfigure(0, weight=1)
        evidence_box.columnconfigure(0, weight=1)

        self.evidence_text = tk.Text(
            evidence_box,
            height=5,
            wrap="word",
            font=("Segoe UI", 9),
            state="disabled",
        )
        self.evidence_text.grid(
            row=0, column=0, sticky="nsew"
        )
        ev_scrollbar = ttk.Scrollbar(
            evidence_box,
            orient="vertical",
            command=self.evidence_text.yview,
        )
        ev_scrollbar.grid(row=0, column=1, sticky="ns")
        self.evidence_text.configure(
            yscrollcommand=ev_scrollbar.set
        )

        raw_box = ttk.LabelFrame(
            side, text="Backend JSON", padding=10
        )
        raw_box.grid(
            row=3, column=0, sticky="nsew"
        )
        raw_box.rowconfigure(0, weight=1)
        raw_box.columnconfigure(0, weight=1)

        self.json_text = tk.Text(
            raw_box,
            height=8,
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
        )
        self.json_text.grid(
            row=0, column=0, sticky="nsew"
        )

        scrollbar = ttk.Scrollbar(
            raw_box,
            orient="vertical",
            command=self.json_text.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.json_text.configure(
            yscrollcommand=scrollbar.set
        )

    # image selection

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Select fundus image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp",
                ),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.set_image(Path(path))

    def _on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        if not paths:
            return

        path = Path(paths[0])
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            messagebox.showerror(
                "Invalid file",
                "Please drop a supported image file.",
            )
            return

        self.set_image(path)

    def set_image(self, path: Path):
        try:
            with Image.open(path) as img:
                self.original_image = img.convert("RGB")
        except Exception as exc:
            messagebox.showerror(
                "Image error",
                f"Could not open image:\n{exc}",
            )
            return

        self.image_path = path
        self.status_var.set(path.name)
        self.drop_hint.place_forget()
        self._clear_results()
        self._refresh_image()

    def _load_samples(self):
        self.sample_list.delete(0, "end")
        self._sample_paths = []

        if not self.sample_dir.exists():
            self.sample_list.insert(
                "end", "No sample directory found"
            )
            return

        paths = sorted(
            p
            for p in self.sample_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not paths:
            self.sample_list.insert(
                "end", "No sample images found"
            )
            return

        self._sample_paths = paths
        for p in paths:
            self.sample_list.insert(
                "end", p.name
            )

    def _select_sample(self, _event=None):
        selection = self.sample_list.curselection()
        if not selection or not self._sample_paths:
            return

        self.set_image(
            self._sample_paths[selection[0]]
        )

    # overlay

    def _refresh_image(self):
        if self.original_image is None:
            return

        canvas_w = max(
            self.canvas.winfo_width(), 500
        )
        canvas_h = max(
            self.canvas.winfo_height(), 400
        )

        img = self.original_image.copy()

        if self.overlay_enabled.get() and self._regions:
            self._draw_regions(img, self._regions)

        scale = min(
            canvas_w / img.width,
            canvas_h / img.height,
            1.0,
        )

        new_size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale)),
        )
        img = img.resize(
            new_size, Image.Resampling.LANCZOS
        )

        self.display_image = img
        self.photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w / 2,
            canvas_h / 2,
            anchor="center",
            image=self.photo,
        )

    def _draw_regions(
        self,
        image: Image.Image,
        regions: list[dict[str, Any]],
    ):
        overlay = Image.new(
            "RGBA", image.size, (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(overlay, "RGBA")

        for region in regions:
            score = float(
                region.get(
                    "score",
                    region.get("importance", 1.0),
                )
            )
            score = max(0.0, min(1.0, score))
            alpha = int(60 + 150 * score)

            if "polygon" in region:
                points = []
                for x, y in region["polygon"]:
                    points.append(
                        self._coord(
                            x, y, image.size, region
                        )
                    )
                if len(points) >= 3:
                    draw.polygon(
                        points,
                        fill=(255, 70, 30, alpha),
                    )
                    draw.line(
                        points + [points[0]],
                        fill=(255, 220, 80, 230),
                        width=2,
                    )
                continue

            if all(
                k in region
                for k in ("x1", "y1", "x2", "y2")
            ):
                x1, y1 = self._coord(
                    region["x1"],
                    region["y1"],
                    image.size,
                    region,
                )
                x2, y2 = self._coord(
                    region["x2"],
                    region["y2"],
                    image.size,
                    region,
                )
            else:
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get(
                    "width",
                    region.get("w", 0),
                )
                h = region.get(
                    "height",
                    region.get("h", 0),
                )

                x1, y1 = self._coord(
                    x, y, image.size, region
                )
                x2, y2 = self._coord(
                    x + w,
                    y + h,
                    image.size,
                    region,
                )

            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(255, 70, 30, alpha),
                outline=(255, 220, 80, 230),
                width=2,
            )

        image.paste(
            overlay, (0, 0), overlay
        )

    @staticmethod
    def _coord(
        x: float,
        y: float,
        image_size: tuple[int, int],
        region: dict[str, Any],
    ):
        if bool(region.get("normalized", False)):
            return (
                float(x) * image_size[0],
                float(y) * image_size[1],
            )
        return float(x), float(y)

    # backend

    def _analyze(self):
        if self.image_path is None:
            messagebox.showwarning(
                "No image",
                "Select or drop a fundus image first.",
            )
            return

        self.status_var.set("Analyzing…")
        self.grade_var.set("…")
        self.confidence_var.set("Confidence: …")
        self.referable_var.set(
            "Referable DR: …"
        )
        self.review_var.set(
            "Human review: …"
        )
        self.explain_time_var.set(
            "Explainability: …"
        )

        threading.Thread(
            target=self._backend_worker,
            args=(self.image_path,),
            daemon=True,
        ).start()

    def _backend_worker(self, path: Path):
        try:
            with path.open("rb") as f:
                response = requests.post(
                    f"{self.backend_url}{self.endpoint}",
                    files={
                        "image": (
                            path.name,
                            f,
                            "application/octet-stream",
                        )
                    },
                    timeout=self.timeout,
                )

            response.raise_for_status()
            data = response.json()

            self.root.after(
                0,
                lambda: self._handle_backend_result(data),
            )

        except requests.RequestException as exc:
            self.root.after(
                0,
                lambda: self._backend_error(
                    f"Backend request failed:\n{exc}"
                ),
            )
        except ValueError as exc:
            self.root.after(
                0,
                lambda: self._backend_error(
                    f"Backend returned invalid JSON:\n{exc}"
                ),
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda: self._backend_error(
                    f"Unexpected error:\n{exc}"
                ),
            )

    def _handle_backend_result(
        self, data: dict[str, Any]
    ):
        self.status_var.set(
            "Analysis complete."
        )

        grade = data.get(
            "dr_grade",
            data.get(
                "grade",
                data.get("prediction", "—"),
            ),
        )
        if isinstance(grade, dict):
            grade = grade.get(
                "grade", "—"
            )

        self.grade_var.set(
            f"Grade {grade}"
        )

        confidence = data.get(
            "confidence",
            data.get(
                "prediction_confidence"
            ),
        )
        if confidence is not None:
            confidence = float(confidence)
            if confidence <= 1:
                confidence *= 100
            self.confidence_var.set(
                f"Confidence: {confidence:.1f}%"
            )
        else:
            self.confidence_var.set(
                "Confidence: —"
            )

        referable = data.get(
            "referable_dr",
            data.get("referable"),
        )
        if (
            referable is None
            and isinstance(
                grade, (int, float)
            )
        ):
            referable = int(grade) >= 2

        if referable is not None:
            self.referable_var.set(
                "Referable DR: "
                + ("YES" if referable else "NO")
            )

        review = data.get(
            "human_review",
            data.get("flagged_for_review"),
        )
        self.review_var.set(
            (
                "Human review: "
                + ("YES" if review else "NO")
            )
            if review is not None
            else "Human review: —"
        )

        explain_time = data.get(
            "explainability_time_ms",
            data.get("gradcam_time_ms"),
        )
        self.explain_time_var.set(
            (
                f"Explainability: "
                f"{float(explain_time):.1f} ms"
            )
            if explain_time is not None
            else "Explainability: —"
        )

        self._regions = self._extract_regions(data)
        self._refresh_image()

        evidence = data.get(
            "evidence",
            data.get("lesions", {}),
        )
        self._set_text(
            self.evidence_text,
            self._format_evidence(evidence),
        )
        self._set_text(
            self.json_text,
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _extract_regions(
        data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        gradcam = data.get(
            "gradcam", {}
        )
        if isinstance(gradcam, dict):
            regions = gradcam.get(
                "regions", []
            )
            if regions:
                return regions

        regions = data.get(
            "regions",
            data.get(
                "gradcam_regions", []
            ),
        )
        return (
            regions
            if isinstance(regions, list)
            else []
        )

    @staticmethod
    def _format_evidence(
        evidence: Any
    ) -> str:
        if not evidence:
            return (
                "No lesion/evidence metadata "
                "returned."
            )

        if isinstance(evidence, dict):
            return "\n".join(
                f"{key.replace('_', ' ').title()}: {value}"
                for key, value in evidence.items()
            )

        return str(evidence)

    @staticmethod
    def _set_text(
        widget: tk.Text,
        text: str,
    ):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _backend_error(self, message: str):
        self.status_var.set(
            "Backend error."
        )
        self.grade_var.set("—")
        messagebox.showerror(
            "Analysis failed",
            message,
        )

    def _clear_results(self):
        self._regions = []
        self.grade_var.set("—")
        self.confidence_var.set(
            "Confidence: —"
        )
        self.referable_var.set(
            "Referable DR: —"
        )
        self.review_var.set(
            "Human review: —"
        )
        self.explain_time_var.set(
            "Explainability: —"
        )
        self._set_text(
            self.evidence_text, ""
        )
        self._set_text(
            self.json_text, ""
        )
        self._refresh_image()


def main():
    config = load_config()

    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = DRScreeningGUI(root, config)

    app.canvas.bind(
        "<Configure>",
        lambda _event: app._refresh_image(),
    )

    root.mainloop()


if __name__ == "__main__":
    main()
