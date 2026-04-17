import os
import pandas as pd
import xlsxwriter
import argparse
from pathlib import Path

def save_images_to_excel(job_folder):
    # Determine the base backup directory
    year = job_folder[:4]
    month = job_folder[4:6]
    
    base_backup_dir = Path(r"D:\Dre\JQ_SPI_02_AI_API\backup")
    job_dir = base_backup_dir / year / month / job_folder
    
    if not job_dir.exists():
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
    
    output_xlsx = job_dir / f"{job_folder}_with_images.xlsx"

    print(f"Processing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create an Excel workbook and a worksheet
    workbook = xlsxwriter.Workbook(str(output_xlsx))
    worksheet = workbook.add_worksheet()

    # Define column widths
    # Col 0-35: CSV data
    # Col 36: Origin Image
    # Col 37: Distance Image
    # Col 38: Anomaly Image
    img_col_width = 60  # Adjust as needed
    
    # Write headers
    headers = list(df.columns) + ["Origin Image", "Distance Image", "Anomaly Image"]
    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header)

    # Image column indices
    origin_col = len(df.columns)
    dist_col = len(df.columns) + 1
    anomaly_col = len(df.columns) + 2

    # Set column widths for images
    worksheet.set_column(origin_col, anomaly_col, img_col_width)

    print("Embedding images...")
    for row_num, (index, row) in enumerate(df.iterrows(), start=1):
        # Write CSV row data
        for col_num, value in enumerate(row):
            worksheet.write(row_num, col_num, str(value) if pd.notna(value) else "")

        img_name = row.get('img_name')
        if not img_name:
            continue

        path_origin = job_dir / img_name
        path_dist = dist_dir / img_name
        path_anomaly = anomaly_dir / f"vis_{img_name}"

        # Set row height to accommodate images (approx 200 pixels)
        worksheet.set_row(row_num, 200)

        # Insert images
        for col, path in [(origin_col, path_origin), (dist_col, path_dist), (anomaly_col, path_anomaly)]:
            if path.exists():
                # Scale image to fit the row/column better
                worksheet.insert_image(row_num, col, str(path), {
                    'x_scale': 1.6,
                    'y_scale': 1.6,
                    'x_offset': 5,
                    'y_offset': 5,
                    'positioning': 1  # Move and size with cells
                })
            else:
                worksheet.write(row_num, col, "Missing")

    workbook.close()
    print(f"\nSuccess! Excel file with embedded images saved to: {output_xlsx}")
    return output_xlsx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save images to an Excel file based on a CSV.")
    parser.add_argument("jobfolder", help="The name of the job folder (e.g., 20260406094033)")
    args = parser.parse_args()
    
    save_images_to_excel(args.jobfolder)
