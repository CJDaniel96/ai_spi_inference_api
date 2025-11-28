import argparse
import re
import pandas as pd
from datetime import datetime

def parse_log_file(log_file_path: str) -> pd.DataFrame:
    """
    Parses a log file to extract job processing metrics.

    Args:
        log_file_path: The path to the log file.

    Returns:
        A pandas DataFrame containing the parsed and calculated data.
    """
    # Regex to capture the key-value pairs from the 'Processing' lines
    log_pattern = re.compile(
        r"Processing .*: "
        r"TargetFolder=(?P<jobfolder>\S+), "
        r"FolderCreated=(?P<FolderCreated>[\d\- :]+), "
        r"CSVCreated=(?P<CSVCreated>[\d\- :]+), "
        r"CSVBackup=(?P<CSVBackup>[\d\- :]+), "
        r"ImageCount=(?P<ImageCount>\d+), "
        r"ProcessEnd=(?P<ProcessEnd>[\d\- :]+), "
        r"TotalTime=(?P<TotalTime>\d+\.\d+)"
    )

    records = []
    print(f"Reading log file: {log_file_path}")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    data = match.groupdict()
                    try:
                        # Convert timestamp strings to datetime objects
                        process_end_dt = datetime.strptime(data['ProcessEnd'], '%Y-%m-%d %H:%M:%S')
                        csv_created_dt = datetime.strptime(data['CSVCreated'], '%Y-%m-%d %H:%M:%S')

                        # Calculate the waiting time in seconds
                        spi_equip_waiting = (process_end_dt - csv_created_dt).total_seconds()

                        records.append({
                            'jobfolder': data['jobfolder'],
                            'TotalTime': float(data['TotalTime']),
                            'spi_equip_waiting_seconds': spi_equip_waiting,
                            'ImageCount': int(data['ImageCount']),
                            'CSVCreated': data['CSVCreated'],
                            'ProcessEnd': data['ProcessEnd'],
                        })
                    except (ValueError, KeyError) as e:
                        print(f"Skipping malformed line due to error: {e}\nLine: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return pd.DataFrame()

    print(f"Successfully parsed {len(records)} records.")
    return pd.DataFrame(records)

def main():
    """Main function to run the log analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze a job log file and output a CSV with performance metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "log_file",
        help="Path to the input log file."
    )
    parser.add_argument(
        "-o", "--output",
        default="log_analysis_results.csv",
        help="Path to the output CSV file."
    )
    args = parser.parse_args()

    df = parse_log_file(args.log_file)

    if not df.empty:
        # Save the DataFrame to a CSV file
        df.to_csv(args.output, index=False)
        print(f"Analysis complete. Results saved to '{args.output}'")
        print("\n--- First 5 rows of the output ---")
        print(df.head())
        print("\n--- Summary Statistics ---")
        print(df.describe())

if __name__ == "__main__":
    main()