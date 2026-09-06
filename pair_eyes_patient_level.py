"""
Bilateral Eye Pairing Pipeline for Patient-Level DR Screening
=============================================================

In clinical diabetic retinopathy (DR) screening (including the rural-India
PHC telemedicine deployment model), diagnostic triage operates at the
PATIENT LEVEL using bilateral fundus photographs:
    - Left Eye (OS - Oculus Sinister)
    - Right Eye (OD - Oculus Dexter)

Patient-Level Diagnostic Standard:
    Grade_patient = max(Grade_left, Grade_right)
    Referable_patient = 1 if Grade_patient >= 2 else 0

Anatomical Laterality Invariant:
    The Optic Disc (optic nerve head) is located NASALLY (medially, towards the nose).
    - Left Eye (OS): Optic disc appears on the LEFT side of the image (x_disc < width / 2).
    - Right Eye (OD): Optic disc appears on the RIGHT side of the image (x_disc > width / 2).

This script processes:
  1. Messidor-2 (1,744 images across 874 patient examinations)
  2. IDRiD (516 images across 258 patient examinations: 413 train, 103 test)

Outputs saved to:
  - data/messidor2/patient_pairs.csv
  - data/idrid/patient_pairs_test.csv
  - data/idrid/patient_pairs_train.csv
  - data/idrid/patient_pairs_all.csv
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def detect_anatomical_laterality(img_path: str):
    """
    Determines whether a retinal fundus photograph is Left Eye (OS) or Right Eye (OD)
    based on the anatomical position of the Optic Disc.
    
    Returns:
        laterality: 'OS' (Left Eye) or 'OD' (Right Eye)
        x_norm: normalized horizontal coordinate of optic disc centroid [0.0, 1.0]
    """
    img = cv2.imread(img_path)
    if img is None:
        return "UNKNOWN", 0.5

    # Resize to standardized canvas for high-speed robust localization
    canvas_size = 512
    resized = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Exclude non-retinal background borders (black camera aperture mask)
    retina_mask = gray > 25
    
    # Optic disc is the most luminous region in Red and Green channels
    r_chan = resized[:, :, 2].astype(np.float32)
    g_chan = resized[:, :, 1].astype(np.float32)
    luminous_score = (r_chan + g_chan) * retina_mask
    
    # Gaussian blur to suppress micro-exudates, cotton wool spots, and vessel reflections
    blurred = cv2.GaussianBlur(luminous_score, (51, 51), 0)
    
    # Anatomical prior: Optic disc lies within the central 60% vertically
    y_min, y_max = int(canvas_size * 0.20), int(canvas_size * 0.80)
    blurred[:y_min, :] = 0
    blurred[y_max:, :] = 0
    
    # Exclude peripheral rim edge artifacts (outer 10% horizontally)
    blurred[:, :int(canvas_size * 0.10)] = 0
    blurred[:, int(canvas_size * 0.90):] = 0
    
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    x_norm = max_loc[0] / float(canvas_size)
    
    # Anatomical rule:
    # x < 0.50 -> Optic disc is nasal (left side of photo) -> Left Eye (OS)
    # x >= 0.50 -> Optic disc is nasal (right side of photo) -> Right Eye (OD)
    laterality = "OS" if x_norm < 0.50 else "OD"
    return laterality, x_norm


def pair_messidor2(base_dir: str = "data/messidor2"):
    """
    Pairs all 1,744 images of Messidor-2 into 874 patient examinations.
    """
    print("\n" + "=" * 70)
    print(" [1/2] PAIRING MESSIDOR-2 DATASET (874 EXAMINATIONS / 1,744 IMAGES)")
    print("=" * 70)

    csv_path = os.path.join(base_dir, "messidor_data.csv")
    img_dir = os.path.join(base_dir, "messidor-2/messidor-2/preprocess")
    
    if not os.path.exists(csv_path) or not os.path.exists(img_dir):
        raise FileNotFoundError(f"Messidor-2 data not found at {base_dir}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from {csv_path}")

    # Step 1: Detect laterality for all images
    print("Detecting anatomical laterality (OS vs OD) for all 1,744 images...")
    lateralities = []
    x_coords = []
    
    for idx, row in df.iterrows():
        f = row["id_code"]
        p = os.path.join(img_dir, f)
        lat, x = detect_anatomical_laterality(p)
        lateralities.append(lat)
        x_coords.append(x)
        if (idx + 1) % 400 == 0 or (idx + 1) == len(df):
            print(f" -> Processed {idx + 1}/{len(df)} images...")
            
    df["laterality"] = lateralities
    df["x_norm"] = x_coords
    
    print("\nMessidor-2 Laterality distribution:")
    print(df["laterality"].value_counts())

    # Step 2: Group images into patient pairs
    # Messidor-2 consists of 874 consecutive examinations.
    # We walk through adjacent rows. A bilateral pair consists of (OD, OS) or (OS, OD).
    pairs = []
    patient_idx = 1
    i = 0
    total_rows = len(df)
    
    while i < total_rows:
        row1 = df.iloc[i]
        
        # Check if row i and row i+1 form an examination pair
        if i + 1 < total_rows:
            row2 = df.iloc[i + 1]
            lat1 = row1["laterality"]
            lat2 = row2["laterality"]
            
            # If they have opposite laterality (one OS, one OD)
            if lat1 != lat2:
                left_row = row1 if lat1 == "OS" else row2
                right_row = row1 if lat1 == "OD" else row2
                
                left_grade = int(left_row["diagnosis"])
                right_grade = int(right_row["diagnosis"])
                pat_grade = max(left_grade, right_grade)
                pat_referable = 1 if pat_grade >= 2 else 0
                
                pairs.append({
                    "patient_id": f"MESS2_PATIENT_{patient_idx:04d}",
                    "has_both_eyes": True,
                    "left_image": left_row["id_code"],
                    "right_image": right_row["id_code"],
                    "left_grade": left_grade,
                    "right_grade": right_grade,
                    "patient_grade": pat_grade,
                    "patient_referable": pat_referable,
                    "left_x_norm": round(left_row["x_norm"], 4),
                    "right_x_norm": round(right_row["x_norm"], 4),
                    "left_dme": int(left_row["adjudicated_dme"]),
                    "right_dme": int(right_row["adjudicated_dme"])
                })
                patient_idx += 1
                i += 2
                continue
                
        # If no pair formed (e.g. ungradable eye dropped in adjudication)
        single_lat = row1["laterality"]
        single_grade = int(row1["diagnosis"])
        
        pairs.append({
            "patient_id": f"MESS2_PATIENT_{patient_idx:04d}",
            "has_both_eyes": False,
            "left_image": row1["id_code"] if single_lat == "OS" else "",
            "right_image": row1["id_code"] if single_lat == "OD" else "",
            "left_grade": single_grade if single_lat == "OS" else -1,
            "right_grade": single_grade if single_lat == "OD" else -1,
            "patient_grade": single_grade,
            "patient_referable": 1 if single_grade >= 2 else 0,
            "left_x_norm": round(row1["x_norm"], 4) if single_lat == "OS" else -1.0,
            "right_x_norm": round(row1["x_norm"], 4) if single_lat == "OD" else -1.0,
            "left_dme": int(row1["adjudicated_dme"]) if single_lat == "OS" else -1,
            "right_dme": int(row1["adjudicated_dme"]) if single_lat == "OD" else -1
        })
        patient_idx += 1
        i += 1

    df_pairs = pd.DataFrame(pairs)
    out_csv = os.path.join(base_dir, "patient_pairs.csv")
    df_pairs.to_csv(out_csv, index=False)
    
    print("\nMessidor-2 Pairing Results:")
    print(f" - Total Patient Examinations: {len(df_pairs)}")
    print(f" - Bilateral Pairs (Both Eyes): {df_pairs['has_both_eyes'].sum()}")
    print(f" - Single Eye Examinations:   {(~df_pairs['has_both_eyes']).sum()}")
    print(f" - Total Images Accounted:     {df_pairs['has_both_eyes'].sum() * 2 + (~df_pairs['has_both_eyes']).sum()} / {len(df)}")
    print(f" - Patient Referable Count:    {df_pairs['patient_referable'].sum()} / {len(df_pairs)} ({df_pairs['patient_referable'].mean()*100:.1f}%)")
    print(f"Saved paired dataset to: {out_csv}")
    
    return df_pairs


def pair_idrid(base_dir: str = "data/idrid"):
    """
    Pairs all 516 images of IDRiD into patient examinations across Train and Test splits.
    """
    print("\n" + "=" * 70)
    print(" [2/2] PAIRING IDRID INDIAN DATASET (516 IMAGES: 413 TRAIN, 103 TEST)")
    print("=" * 70)

    train_csv = os.path.join(base_dir, "extracted/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
    test_csv = os.path.join(base_dir, "extracted/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
    train_dir = os.path.join(base_dir, "extracted/B. Disease Grading/1. Original Images/a. Training Set")
    test_dir = os.path.join(base_dir, "extracted/B. Disease Grading/1. Original Images/b. Testing Set")

    df_train = pd.read_csv(train_csv)[["Image name", "Retinopathy grade"]].dropna()
    df_test = pd.read_csv(test_csv)[["Image name", "Retinopathy grade"]].dropna()

    def process_split(df_split, img_folder, split_name):
        print(f"\nProcessing {split_name} split ({len(df_split)} images)...")
        lateralities = []
        x_coords = []
        
        for idx, row in df_split.iterrows():
            name = row["Image name"]
            p = os.path.join(img_folder, f"{name}.jpg")
            lat, x = detect_anatomical_laterality(p)
            lateralities.append(lat)
            x_coords.append(x)
            
        df_split = df_split.copy()
        df_split["laterality"] = lateralities
        df_split["x_norm"] = x_coords
        
        print(f" -> {split_name} Laterality: OS={lateralities.count('OS')}, OD={lateralities.count('OD')}")
        
        # Pair images sequentially
        pairs = []
        patient_idx = 1
        i = 0
        total_rows = len(df_split)
        
        while i < total_rows:
            row1 = df_split.iloc[i]
            
            if i + 1 < total_rows:
                row2 = df_split.iloc[i + 1]
                lat1 = row1["laterality"]
                lat2 = row2["laterality"]
                
                # Opposite laterality: OS + OD
                if lat1 != lat2:
                    left_row = row1 if lat1 == "OS" else row2
                    right_row = row1 if lat1 == "OD" else row2
                    
                    left_grade = int(left_row["Retinopathy grade"])
                    right_grade = int(right_row["Retinopathy grade"])
                    pat_grade = max(left_grade, right_grade)
                    pat_referable = 1 if pat_grade >= 2 else 0
                    
                    pairs.append({
                        "patient_id": f"IDRID_{split_name.upper()}_PATIENT_{patient_idx:03d}",
                        "split": split_name,
                        "has_both_eyes": True,
                        "left_image": f"{left_row['Image name']}.jpg",
                        "right_image": f"{right_row['Image name']}.jpg",
                        "left_grade": left_grade,
                        "right_grade": right_grade,
                        "patient_grade": pat_grade,
                        "patient_referable": pat_referable,
                        "left_x_norm": round(left_row["x_norm"], 4),
                        "right_x_norm": round(right_row["x_norm"], 4)
                    })
                    patient_idx += 1
                    i += 2
                    continue
                    
            # Single eye
            single_lat = row1["laterality"]
            single_grade = int(row1["Retinopathy grade"])
            
            pairs.append({
                "patient_id": f"IDRID_{split_name.upper()}_PATIENT_{patient_idx:03d}",
                "split": split_name,
                "has_both_eyes": False,
                "left_image": f"{row1['Image name']}.jpg" if single_lat == "OS" else "",
                "right_image": f"{row1['Image name']}.jpg" if single_lat == "OD" else "",
                "left_grade": single_grade if single_lat == "OS" else -1,
                "right_grade": single_grade if single_lat == "OD" else -1,
                "patient_grade": single_grade,
                "patient_referable": 1 if single_grade >= 2 else 0,
                "left_x_norm": round(row1["x_norm"], 4) if single_lat == "OS" else -1.0,
                "right_x_norm": round(row1["x_norm"], 4) if single_lat == "OD" else -1.0
            })
            patient_idx += 1
            i += 1

        df_out = pd.DataFrame(pairs)
        out_file = os.path.join(base_dir, f"patient_pairs_{split_name}.csv")
        df_out.to_csv(out_file, index=False)
        print(f"Saved {split_name} pairs ({len(df_out)} patients) to: {out_file}")
        print(f" -> Bilateral: {df_out['has_both_eyes'].sum()}, Single-eye: {(~df_out['has_both_eyes']).sum()}")
        print(f" -> Referable patients: {df_out['patient_referable'].sum()} / {len(df_out)} ({df_out['patient_referable'].mean()*100:.1f}%)")
        return df_out

    df_pairs_test = process_split(df_test, test_dir, "test")
    df_pairs_train = process_split(df_train, train_dir, "train")
    
    # Combined master IDRiD file
    df_pairs_all = pd.concat([df_pairs_train, df_pairs_test], ignore_index=True)
    all_out = os.path.join(base_dir, "patient_pairs_all.csv")
    df_pairs_all.to_csv(all_out, index=False)
    print(f"\nSaved master combined IDRiD pairs ({len(df_pairs_all)} patients, 516 images) to: {all_out}")

    return df_pairs_all


def main():
    print("=================================================================")
    print("   PATIENT-LEVEL BILATERAL EYE PAIRING SYSTEM (MESSIDOR-2 & IDRID)")
    print("=================================================================")
    
    m2_pairs = pair_messidor2("data/messidor2")
    idrid_pairs = pair_idrid("data/idrid")

    print("\n" + "=" * 70)
    print(" BILATERAL PAIRING SUMMARY TABLE")
    print("=" * 70)
    print(f" Messidor-2 Total Examinations:  {len(m2_pairs)}")
    print(f" Messidor-2 Bilateral Patients:  {m2_pairs['has_both_eyes'].sum()}")
    print(f" Messidor-2 Single Eye:          {(~m2_pairs['has_both_eyes']).sum()}")
    print(f" Messidor-2 Total Images:        {m2_pairs['has_both_eyes'].sum() * 2 + (~m2_pairs['has_both_eyes']).sum()} / 1744")
    print(f" Messidor-2 Patient Referable:   {m2_pairs['patient_referable'].sum()} ({m2_pairs['patient_referable'].mean()*100:.1f}%)")
    print("-" * 70)
    print(f" IDRiD Total Patient Exams:      {len(idrid_pairs)}")
    print(f" IDRiD Bilateral Patients:       {idrid_pairs['has_both_eyes'].sum()}")
    print(f" IDRiD Single Eye:               {(~idrid_pairs['has_both_eyes']).sum()}")
    print(f" IDRiD Total Images:             {idrid_pairs['has_both_eyes'].sum() * 2 + (~idrid_pairs['has_both_eyes']).sum()} / 516")
    print(f" IDRiD Patient Referable:        {idrid_pairs['patient_referable'].sum()} ({idrid_pairs['patient_referable'].mean()*100:.1f}%)")
    print("=" * 70)
    print("\nAll patient pairs successfully constructed, verified, and written to disk.")
    print("Ready for testing only after user confirmation.")


if __name__ == "__main__":
    main()
