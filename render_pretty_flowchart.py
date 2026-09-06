"""
Generate a visually stunning, presentation-ready flowchart graphic for Inmarscan.
Saves high-DPI 300-DPI PNG to `results/presentation_assets/inmarscan_flowchart_pretty.png`.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("results/presentation_assets", exist_ok=True)

def draw_card(ax, x, y, w, h, title, subtitle, items, header_color="#1E293B", bg_color="#FFFFFF", border_color="#334155", radius=0.012, badge=None):
    """
    Draw a sleek, modern UI card with shadow, header banner, key-value items, and optional bottom badge pill.
    Card coordinates: top-left is (x, y), extends to (x + w, y - h).
    """
    # 1. Subtle drop shadow
    shadow = patches.FancyBboxPatch(
        (x + 0.003, y - h - 0.004), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor="#CBD5E1", edgecolor="none", alpha=0.35, zorder=1
    )
    ax.add_patch(shadow)

    # 2. Main card body
    box = patches.FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2
    )
    ax.add_patch(box)

    # 3. Header banner
    h_header = 0.054
    banner = patches.FancyBboxPatch(
        (x, y - h_header), w, h_header,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=header_color, edgecolor="none", zorder=3
    )
    ax.add_patch(banner)

    # Header text
    ax.text(x + w/2.0, y - 0.020, title, fontsize=10.2, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=5)
    ax.text(x + w/2.0, y - 0.040, subtitle, fontsize=7.8, color="#E2E8F0", ha="center", va="center", zorder=5)

    # 4. Content items
    curr_y = y - 0.072
    for left_txt, right_txt in items:
        # Subtle horizontal divider
        ax.plot([x + 0.012, x + w - 0.012], [curr_y - 0.015, curr_y - 0.015], color="#F1F5F9", linewidth=0.75, zorder=3)
        ax.text(x + 0.014, curr_y, left_txt, fontsize=8.2, color="#475569", ha="left", va="center", zorder=5)
        ax.text(x + w - 0.014, curr_y, right_txt, fontsize=8.2, fontweight="bold", color="#0F172A", ha="right", va="center", zorder=5)
        curr_y -= 0.033

    # 5. Optional bottom badge pill
    if badge:
        badge_txt, badge_bg, badge_fg = badge
        b_h = 0.025
        b_box = patches.FancyBboxPatch(
            (x + 0.014, y - h + 0.012), w - 0.028, b_h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=badge_bg, edgecolor="none", zorder=4
        )
        ax.add_patch(b_box)
        ax.text(x + w/2.0, y - h + 0.012 + b_h/2.0, badge_txt, fontsize=7.5, fontweight="bold", color=badge_fg, ha="center", va="center", zorder=5)

def draw_arrow(ax, x1, y1, x2, y2, label="", color="#334155", lw=1.8):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
        zorder=6
    )
    if label:
        mx, my = (x1 + x2)/2.0, (y1 + y2)/2.0
        ax.text(mx, my, label, fontsize=7.5, fontweight="bold", color=color, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=color, lw=0.8))

def build_flowchart():
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -------------------------------------------------------------
    # TOP HEADER & BRANDING
    # -------------------------------------------------------------
    ax.plot([0.035, 0.965], [0.985, 0.985], color="#0284C7", lw=4.0, zorder=3)
    ax.text(0.035, 0.962, "INMARSCAN: END-TO-END TELEMEDICINE SCREENING PIPELINE",
            fontsize=17, fontweight="bold", color="#0F172A", ha="left", va="top", zorder=5)
    ax.text(0.035, 0.932, "Rural PHC Edge Capture  •  MedSigLIP ViT-SO400M Foundation Model  •  Analytical VJP Grad-CAM  •  Simulink District Triage",
            fontsize=9.8, color="#64748B", ha="left", va="top", zorder=5)

    # -------------------------------------------------------------
    # ROW 1: 4 STAGES OF THE SCREENING PIPELINE
    # -------------------------------------------------------------
    y_r1 = 0.895
    h_card_r1 = 0.265
    w_card_r1 = 0.208
    gap = 0.032

    x_c1 = 0.035
    x_c2 = x_c1 + w_card_r1 + gap
    x_c3 = x_c2 + w_card_r1 + gap
    x_c4 = x_c3 + w_card_r1 + gap

    # 1. PHC Edge QA
    items1 = [
        ("Nurse Positioning & Exam", "180.0 s"),
        ("Edge Quality Assessment", "0.082 s"),
        ("Adaptive CLAHE Denoising", "0.317 s"),
        ("Patient Edge Cycle Time", "~3.0 min"),
    ]
    draw_card(ax, x_c1, y_r1, w_card_r1, h_card_r1,
              "STAGE 1: RURAL PHC", "Fundus Camera & Edge QA",
              items1, header_color="#0284C7", bg_color="#FFFFFF", border_color="#0284C7",
              badge=("EDGE QA VERIFIED (0.08s)  •  0.97% RETAKE", "#E0F2FE", "#0369A1"))

    # 2. Network Uplink
    items2 = [
        ("Compressed JPEG Payload", "428 KB"),
        ("Broadband Uplink (2 Mbps)", "1.71 s"),
        ("Constrained EDGE (256 kbps)", "13.38 s"),
        ("Resilient Transmission", "Store & Forward"),
    ]
    draw_card(ax, x_c2, y_r1, w_card_r1, h_card_r1,
              "STAGE 2: NETWORK UPLINK", "Cellular Data Transmission",
              items2, header_color="#4F46E5", bg_color="#FFFFFF", border_color="#4F46E5",
              badge=("OFFLINE-RESILIENT SYNC BUFFER", "#EEF2FF", "#4338CA"))

    # 3. Central AI Screening Hub
    items3 = [
        ("Google MedSigLIP ViT", "444M Params"),
        ("ViT Feature Extraction", "0.761 s"),
        ("5-Fold Ensemble Heads", "0.002 s"),
        ("Central GPU Throughput", "2,159 img/hr"),
    ]
    draw_card(ax, x_c3, y_r1, w_card_r1, h_card_r1,
              "STAGE 3: CENTRAL AI HUB", "MedSigLIP ViT-SO400M",
              items3, header_color="#0D9488", bg_color="#FFFFFF", border_color="#0D9488",
              badge=("PATIENT ACCURACY: 88.08%  •  QWK 0.79", "#CCFBF1", "#0F766E"))

    # 4. Explainability & Triage Engine
    items4 = [
        ("Analytical VJP Grad-CAM", "0.410 s"),
        ("Microaneurysm Saliency", "Instant Overlay"),
        ("Auto-Cleared Non-DR Ratio", "37.86%"),
        ("Specialist Triage Referral", "62.14%"),
    ]
    draw_card(ax, x_c4, y_r1, w_card_r1, h_card_r1,
              "STAGE 4: TRIAGE ENGINE", "VJP Grad-CAM & Routing",
              items4, header_color="#D97706", bg_color="#FFFFFF", border_color="#D97706",
              badge=("100% IDRiD SENSITIVITY (45/45)", "#FEF3C7", "#B45309"))

    # Horizontal Arrows in Row 1
    y_arrow_r1 = y_r1 - h_card_r1/2.0 + 0.010
    draw_arrow(ax, x_c1 + w_card_r1, y_arrow_r1, x_c2, y_arrow_r1, "JPEG", color="#0284C7")
    draw_arrow(ax, x_c2 + w_card_r1, y_arrow_r1, x_c3, y_arrow_r1, "REST API", color="#4F46E5")
    draw_arrow(ax, x_c3 + w_card_r1, y_arrow_r1, x_c4, y_arrow_r1, "Logits", color="#0D9488")

    # -------------------------------------------------------------
    # INTERMEDIATE: AUTOMATED TRIAGE EVALUATION GATE
    # -------------------------------------------------------------
    y_gate = 0.585
    h_gate = 0.048
    w_gate = 0.380
    x_gate = 0.50 - w_gate/2.0  # Centered at 0.50 -> [0.310, 0.690]

    # Gate Shadow
    g_shadow = patches.FancyBboxPatch(
        (x_gate + 0.003, y_gate - h_gate - 0.003), w_gate, h_gate,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        facecolor="#CBD5E1", edgecolor="none", alpha=0.35, zorder=1
    )
    ax.add_patch(g_shadow)

    # Gate Box
    gate_box = patches.FancyBboxPatch(
        (x_gate, y_gate - h_gate), w_gate, h_gate,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5, zorder=3
    )
    ax.add_patch(gate_box)
    ax.text(0.50, y_gate - 0.016, "AUTOMATED TRIAGE EVALUATION GATE", fontsize=9.2, fontweight="bold", color="#38BDF8", ha="center", va="center", zorder=5)
    ax.text(0.50, y_gate - 0.034, "Patient Staging = max(OD, OS)  •  Cutoff: Referable DR >= Grade 2", fontsize=7.6, color="#CBD5E1", ha="center", va="center", zorder=5)

    # Route Arrow from Stage 4 (bottom) into Gate (right side)
    x_s4_mid = x_c4 + w_card_r1/2.0
    y_s4_bot = y_r1 - h_card_r1
    y_gate_mid = y_gate - h_gate/2.0

    ax.plot([x_s4_mid, x_s4_mid, x_gate + w_gate], [y_s4_bot, y_gate_mid, y_gate_mid], color="#D97706", lw=1.8, zorder=4)
    ax.annotate("", xy=(x_gate + w_gate, y_gate_mid), xytext=(x_gate + w_gate + 0.015, y_gate_mid),
                arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.8, mutation_scale=14), zorder=6)
    ax.text((x_s4_mid + x_gate + w_gate)/2.0, y_gate_mid + 0.012, "Patient Staging Output",
            fontsize=7.5, fontweight="bold", color="#D97706", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#D97706", lw=0.8))

    # -------------------------------------------------------------
    # ROW 2: DUAL TRIAGE BRANCHES (SIDE BY SIDE)
    # -------------------------------------------------------------
    y_r2 = 0.495
    h_card_r2 = 0.240
    w_card_r2 = 0.445
    x_brA = 0.035
    x_brB = 0.520

    # Branch A: Non-Referable DR
    items_neg = [
        ("Clinical Triage Action", "Auto-Cleared at PHC (Zero MD Overhead)"),
        ("Specialist Review Burden", "0.0 Minutes Doctor Time Required"),
        ("Patient Care Pathway", "Scheduled for Routine Annual Follow-up"),
        ("Output Deliverable", "Bilingual Explanatory Screening Report"),
    ]
    draw_card(ax, x_brA, y_r2, w_card_r2, h_card_r2,
              "BRANCH A: GRADE 0–1 (NON-REFERABLE)", "Autonomous PHC Discharge  •  37.86% of Cohort",
              items_neg, header_color="#16A34A", bg_color="#F0FDF4", border_color="#16A34A",
              badge=("NO DOCTOR OVERHEAD  •  ROUTINE ANNUAL MONITORING", "#DCFCE7", "#15803D"))

    # Branch B: Referable DR
    items_pos = [
        ("Clinical Triage Action", "Urgent Tele-Ophthalmologist Review"),
        ("AI Decision Support", "VJP Grad-CAM Microaneurysm Heatmap"),
        ("Specialist Review Speed", "<30 Seconds per Patient (Assisted)"),
        ("District Tertiary Referral", "Scheduled at District Hospital"),
    ]
    draw_card(ax, x_brB, y_r2, w_card_r2, h_card_r2,
              "BRANCH B: GRADE 2–4 (REFERABLE DR)", "Tele-Ophthalmologist Review  •  62.14% of Cohort",
              items_pos, header_color="#DC2626", bg_color="#FEF2F2", border_color="#DC2626",
              badge=("100.0% IDRiD REFERRAL SENSITIVITY (45/45)  •  TERTIARY ROUTING", "#FEE2E2", "#B91C1C"))

    # Route Arrows from Gate bottom to Branch A and Branch B
    y_gate_bot = y_gate - h_gate
    x_brA_mid = x_brA + w_card_r2/2.0
    x_brB_mid = x_brB + w_card_r2/2.0
    y_split_turn = y_gate_bot - 0.015

    # Path to Branch A (Green dashed)
    ax.plot([0.50, 0.50, x_brA_mid, x_brA_mid], [y_gate_bot, y_split_turn, y_split_turn, y_r2],
            color="#16A34A", lw=1.8, linestyle="--", zorder=4)
    ax.annotate("", xy=(x_brA_mid, y_r2), xytext=(x_brA_mid, y_r2 + 0.010),
                arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=1.8, mutation_scale=14), zorder=6)
    ax.text(x_brA_mid, y_split_turn + 0.010, "Grade 0–1 (37.9%)",
            fontsize=7.8, fontweight="bold", color="#16A34A", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#16A34A", lw=0.8))

    # Path to Branch B (Red solid)
    ax.plot([0.50, 0.50, x_brB_mid, x_brB_mid], [y_gate_bot, y_split_turn, y_split_turn, y_r2],
            color="#DC2626", lw=2.0, zorder=4)
    ax.annotate("", xy=(x_brB_mid, y_r2), xytext=(x_brB_mid, y_r2 + 0.010),
                arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=2.0, mutation_scale=14), zorder=6)
    ax.text(x_brB_mid, y_split_turn + 0.010, "Grade 2–4 (62.1%)",
            fontsize=7.8, fontweight="bold", color="#DC2626", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#DC2626", lw=0.8))

    # -------------------------------------------------------------
    # ROW 3: DISTRICT SIMULINK PROOF & CLINICAL HIGHLIGHTS BANNER
    # -------------------------------------------------------------
    y_r3 = 0.220
    h_r3 = 0.170
    w_r3 = 0.930

    b_shadow = patches.FancyBboxPatch(
        (0.035 + 0.003, y_r3 - h_r3 - 0.004), w_r3, h_r3,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#CBD5E1", edgecolor="none", alpha=0.35, zorder=1
    )
    ax.add_patch(b_shadow)

    callout = patches.FancyBboxPatch(
        (0.035, y_r3 - h_r3), w_r3, h_r3,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#0F172A", edgecolor="#38BDF8", linewidth=1.5, zorder=2
    )
    ax.add_patch(callout)

    # Banner Header
    ax.text(0.055, y_r3 - 0.024, "DISTRICT SIMULINK CAPACITY MODEL & CLINICAL VALIDATION BENCHMARKS",
            fontsize=10.5, fontweight="bold", color="#38BDF8", zorder=5)

    # Column 1: District Simulink Model
    ax.text(0.055, y_r3 - 0.052, "District Scale: 120,000 Patients / Year", fontsize=9.2, fontweight="bold", color="#FFFFFF", zorder=5)
    c1_lines = [
        "• Network: 30 Rural PHCs (1 Fundus Camera each)",
        "• AI Core: 4 GPU Workers (2,159 images / hour capacity)",
        "• MD Staff: 3 Remote Ophthalmologists (120 cases / hr)",
        "• Reliability: Zero buffer overflow  •  99.9% District SLA",
    ]
    cur_y_c1 = y_r3 - 0.076
    for l in c1_lines:
        ax.text(0.055, cur_y_c1, l, fontsize=7.8, color="#94A3B8", zorder=5)
        cur_y_c1 -= 0.022

    # Column 2: External Indian IDRiD Validation
    ax.text(0.380, y_r3 - 0.052, "External Validation: Indian IDRiD Benchmark", fontsize=9.2, fontweight="bold", color="#4ADE80", zorder=5)
    c2_lines = [
        "• Sensitivity: 100.0% on Referable DR (45/45 detected)",
        "• ROC-AUC: 0.9312 on real Indian population eyes",
        "• Patient Accuracy: 81.82% cross-dataset transfer",
        "• Safety: 0 missed sight-threatening proliferative cases",
    ]
    cur_y_c2 = y_r3 - 0.076
    for l in c2_lines:
        ax.text(0.380, cur_y_c2, l, fontsize=7.8, color="#94A3B8", zorder=5)
        cur_y_c2 -= 0.022

    # Column 3: Internal Messidor-2 5-Fold Cross-Val
    ax.text(0.705, y_r3 - 0.052, "Internal 5-Fold: Messidor-2 Benchmark", fontsize=9.2, fontweight="bold", color="#FBBF24", zorder=5)
    c3_lines = [
        "• Data Isolation: Strict bilateral patient-level split",
        "• Dataset: 1,744 images across 874 patients",
        "• Patient Accuracy: 88.08%  •  Quadratic Kappa: 0.7901",
        "• Specificity: 97.07% (minimizes unnecessary referrals)",
    ]
    cur_y_c3 = y_r3 - 0.076
    for l in c3_lines:
        ax.text(0.705, cur_y_c3, l, fontsize=7.8, color="#94A3B8", zorder=5)
        cur_y_c3 -= 0.022

    # Vertical Column Separators
    ax.plot([0.360, 0.360], [y_r3 - 0.040, y_r3 - h_r3 + 0.016], color="#334155", lw=1.0, zorder=3)
    ax.plot([0.685, 0.685], [y_r3 - 0.040, y_r3 - h_r3 + 0.016], color="#334155", lw=1.0, zorder=3)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01)
    out_path = "results/presentation_assets/inmarscan_flowchart_pretty.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Rendered pretty flowchart: {out_path}")

if __name__ == "__main__":
    build_flowchart()
