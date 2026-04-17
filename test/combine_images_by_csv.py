import os
import pandas as pd
import cv2
import numpy as np
import argparse
from pathlib import Path

def combine_images_by_csv():
    # Determine the base backup directory
    # Based on the user's context, the structure is backup/YYYY/MM/job_folder
    # We extract YYYY and MM from the job_folder (YYYYMMDDHHMMSS)
    job_folder = "20260406094033"
    year = job_folder[:4]
    month = job_folder[4:6]
    
    base_backup_dir = Path(r"D:\Dre\JQ_SPI_02_AI_API\backup")
    job_dir = base_backup_dir / year / month / job_folder
    
    if not job_dir.exists():
        # Try alternate path if not found in YYYY/MM structure
        job_dir = base_backup_dir / job_folder
        if not job_dir.exists():
            print(f"Error: Job folder {job_folder} not found in {base_backup_dir}")
            return

    csv_path = job_dir / f"{job_folder}_processed.csv"
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Define sub-directories
    dist_dir = job_dir / "distance_visualized_results"
    anomaly_dir = job_dir / "anomaly_heatmaps"
    output_dir = job_dir / "combined_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "img_name" not in df.columns:
        print("Error: 'img_name' column not found in CSV.")
        return

    combined_paths = []

    for index, row in df.iterrows():
        img_name = row['img_name']
        
        path_origin = job_dir / img_name
        path_dist = dist_dir / img_name
        path_anomaly = anomaly_dir / img_name
        
        # Load images
        img_origin = cv2.imread(str(path_origin))
        img_dist = cv2.imread(str(path_dist))
        img_anomaly = cv2.imread(str(path_anomaly))
        
        if img_origin is None or img_dist is None or img_anomaly is None:
            print(f"  Warning: One or more images missing for {img_name}")
            combined_paths.append("")
            continue

        # Get the heights and widths
        h1, w1 = img_origin.shape[:2]
        h2, w2 = img_dist.shape[:2]
        h3, w3 = img_anomaly.shape[:2]

        # Use the maximum height for all images
        max_h = max(h1, h2, h3)
        
        # Function to pad images to the same height
        def pad_image(img, target_h):
            h, w = img.shape[:2]
            if h == target_h:
                return img
            pad_bottom = target_h - h
            return cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        img_origin_pad = pad_image(img_origin, max_h)
        img_dist_pad = pad_image(img_dist, max_h)
        img_anomaly_pad = pad_image(img_anomaly, max_h)

        # Concatenate horizontally
        # [Origin | Distance | Anomaly]
        combined = np.hstack((img_origin_pad, img_dist_pad, img_anomaly_pad))
        
        output_filename = f"combined_{img_name}"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), combined)
        
        combined_paths.append(str(output_path))
        print(f"  Saved: {output_filename}")

    # Add the new column and save the new CSV
    df['combined_img_path'] = combined_paths
    output_csv_path = job_dir / f"{job_folder}_combined_processed.csv"
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nSuccess! Combined images saved to: {output_dir}")
    print(f"Resulting CSV saved to: {output_csv_path}")
    return output_csv_path

if __name__ == "__main__":
    combine_images_by_csv()
