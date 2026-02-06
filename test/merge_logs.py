import pandas as pd
import os
import argparse

def merge_log_files(left_file: str, right_file: str, output_file: str):
    """
    Merges two log CSV files based on a job folder identifier.

    The script performs a left merge, using the first file as the left table.
    It extracts a common key from a full path in the right table to match
    the key in the left table.

    Args:
        left_file (str): Path to the left CSV file (e.g., log_analysis_results.csv).
        right_file (str): Path to the right CSV file (e.g., log_02.csv).
        output_file (str): Path to save the merged CSV file.
    """
    print(f"Reading left CSV: {left_file}")
    try:
        df_left = pd.read_csv(left_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{left_file}'. Aborting.")
        return

    print(f"Reading right CSV: {right_file}")
    try:
        df_right = pd.read_csv(right_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{right_file}'. Aborting.")
        return

    # --- Prepare merge keys ---
    # The key in df_left is 'jobfolder' (e.g., '20250918093842')
    # The key in df_right is in 'job_folder' (e.g., 'D:\\...\\20250918093842')

    # Ensure the merge key in the left DataFrame is a string to prevent dtype mismatch.
    if 'jobfolder' in df_left.columns:
        df_left['jobfolder'] = df_left['jobfolder'].astype(str)

    print("Extracting merge key from the right DataFrame ('job_folder' column)...")
    # Create a new column 'jobfolder' in the right df by extracting the basename of the path
    df_right['jobfolder'] = df_right['job_folder'].apply(lambda path: os.path.basename(str(path).strip()))

    # --- Perform the merge ---
    print(f"Performing left merge on 'jobfolder' column...")
    # Use a left merge to keep all records from log_analysis_results.csv
    merged_df = pd.merge(df_left, df_right, on='jobfolder', how='left')

    # --- Post-merge processing ---
    print("Performing post-merge timestamp conversions and calculations...")

    # Define timestamp columns to process
    timestamp_cols = ['ProcessEnd', 'CSVCreated', 'request_start_at']
    # Filter to only include columns that exist in the merged dataframe
    existing_ts_cols = [col for col in timestamp_cols if col in merged_df.columns]

    # Convert timestamp columns to datetime objects for calculation
    for col in existing_ts_cols:
        merged_df[col] = pd.to_datetime(merged_df[col], errors='coerce')

    # --- Filter by date ---
    if 'CSVCreated' in merged_df.columns:
        print("Filtering rows where CSVCreated is after '2025-11-14 09:30:00'...")
        # Perform the filter. Rows with NaT in CSVCreated will be dropped.
        filter_date = pd.to_datetime('2026-01-14 09:30:00')
        merged_df = merged_df[merged_df['CSVCreated'] > filter_date].copy()

    # Calculate '02_scan_time_ms'
    if 'request_start_at' in merged_df.columns and 'ProcessEnd' in merged_df.columns:
        print("Calculating '02_scan_time_ms' column...")
        # Before subtraction, we must handle timezone differences.
        # 'request_start_at' is timezone-aware, 'ProcessEnd' is naive.
        # We will convert 'request_start_at' to naive by removing timezone info.
        request_start_naive = merged_df['request_start_at'].dt.tz_localize(None)

        # Calculate timedelta and convert to total milliseconds
        time_diff = (request_start_naive - merged_df['ProcessEnd'])
        merged_df['02_scan_time_ms'] = time_diff.dt.total_seconds() * 1000
    
    # --- Calculate combined total time ---
    print("Calculating '01_02_total_s' column...")
    # Sum of TotalTime (s) + 02_scan_time_ms (ms -> s) + request_latency_ms (ms -> s)
    # Ensure columns are numeric, coercing errors to NaN
    total_time_s = pd.to_numeric(merged_df['TotalTime'], errors='coerce')
    scan_time_s = pd.to_numeric(merged_df.get('02_scan_time_ms'), errors='coerce') / 1000
    latency_s = pd.to_numeric(merged_df.get('request_latency_ms'), errors='coerce') / 1000
    
    # Use .add() to handle NaNs gracefully during summation
    merged_df['01_02_total_s'] = total_time_s.add(scan_time_s, fill_value=0).add(latency_s, fill_value=0)

    # --- Convert all ms columns to seconds for the final report ---
    print("Converting millisecond columns to seconds...")
    ms_cols_to_convert = [
        '02_scan_time_ms',
        'request_latency_ms',
        'anomaly_request_ms',
        'paste_request_ms',
        'distance_request_ms'
    ]
    for col in ms_cols_to_convert:
        if col in merged_df.columns:
            # Ensure column is numeric before division
            numeric_col = pd.to_numeric(merged_df[col], errors='coerce')
            merged_df[col] = numeric_col / 1000

    # Convert timestamp columns to the desired string format for the output file
    print(f"Formatting timestamp columns to 'YYYY-MM-DD HH:MM:SS.ffffff'...")
    for col in existing_ts_cols:
        # Use .dt accessor to format, NaT values will become None/NaN which is handled by to_csv
        merged_df[col] = merged_df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # --- Add single quote to jobfolder for Excel readability ---
    print("Prepending single quote to 'jobfolder' for Excel readability...")
    if 'jobfolder' in merged_df.columns:
        merged_df['jobfolder'] = "'" + merged_df['jobfolder'].astype(str)

    # --- Rename and Reorder columns for final output ---
    print("Renaming and reordering columns for the output file...")
    column_rename_map = {
        'jobfolder': 'jobfolder',
        'ImageCount': '影像數量',
        'TotalTime': '01花費時間(s)',
        'request_latency_ms': '02花費時間(s)',
        '02_scan_time_ms': '02掃檔花費時間(s)',
        '01_02_total_s': '01&02花費時間(s)',
        'anomaly_request_ms': '異常偵測花費時間(s)',
        'paste_request_ms': '錫膏計算花費時間(s)',
        'distance_request_ms': '錫膏距離計算花費時間(s)',
        'CSVCreated': '01_start_at',
        'ProcessEnd': '01_end_at',
        'request_start_at': '02_request_start_at',
        'request_end_at': '02_request_end_at',
        'csv_transfer': 'CSV傳輸花費時間'
    }
    merged_df.rename(columns=column_rename_map, inplace=True)

    # Define the final order based on the new column names
    final_column_order = [
        'jobfolder', '影像數量', '01花費時間(s)', 'CSV傳輸花費時間', '02花費時間(s)',
        '02掃檔花費時間(s)', '01&02花費時間(s)', '異常偵測花費時間(s)',
        '錫膏計算花費時間(s)', '錫膏距離計算花費時間(s)', '01_start_at',
        '01_end_at', '02_request_start_at', '02_request_end_at'
    ]
    # Filter the DataFrame to only include and order the specified columns
    merged_df = merged_df[final_column_order]

    # --- Save the result ---
    print(f"Saving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("\nMerge complete!")
    print(f"First 5 rows of the merged file:")
    print(merged_df.head())

if __name__ == "__main__":
    # Assuming the script is run from the project root directory
    # The script is in /test, so data is in ../data
    merge_log_files('log_data/log_analysis_results.csv', 'log_data/02.csv', 'log_data/log_analysis_report_tw.csv')