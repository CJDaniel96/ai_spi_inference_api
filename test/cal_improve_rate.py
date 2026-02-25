import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Mode Selection
# MODE = "default"        # Original: Excludes "_processed", calculates Pass/Fail Improvement Rate
MODE = "read_processed" # New: Includes only "_processed", calculates Defect Type %

def plot_improvement_data(result_df, mode, target_defects=None):
    """
    Generates and displays a Plotly chart based on the processed results.
    """
    print("\nGenerating Plotly chart...")

    # We use make_subplots for dual axis if needed, or manual layout. 
    # Manual layout is often simpler for quick dual-axis.
    fig = go.Figure()

    if mode == "default":
        # Create text labels: Round to 1 decimal place and add '%'
        label_text = result_df['Improve Rate %'].round(1).astype(str) + '%'

        fig.add_trace(go.Scatter(
            x=result_df['Date'],
            y=result_df['Improve Rate %'],
            mode='lines+markers+text',
            name='improved rate',
            text=label_text,
            textposition="top center",
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        y_axis_title = "Improved Rate%"
        y_range = [0, 110]
        title_text = "SPI+AI Improved Rate (J15)"

        # --- Calculate Overall Average ---
        total_pass_all = result_df['Pass Count'].sum()
        total_items_all = result_df['Total'].sum()
        if total_items_all > 0:
            overall_avg = (total_pass_all / total_items_all) * 100
        else:
            overall_avg = 0.0
            
        annotation_text = f"<b>Average Improved Rate: {overall_avg:.1f}%</b>"
        
        fig.update_layout(
            yaxis=dict(title=y_axis_title, range=y_range),
            xaxis_title="Date"
        )
        
    elif mode == "read_processed":
        # --- Stacked Bars (Job Row Counts) ---
        fig.add_trace(go.Bar(
            x=result_df['Date'],
            y=result_df['Within_Thres'],
            name='Within Thres (len < 61)',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=result_df['Date'],
            y=result_df['Over_Thres'],
            name='Over Thres (len >= 61)',
            marker_color='orange'
        ))

        # --- Lines (Defect Rates) ---
        if target_defects:
            for d in target_defects:
                fig.add_trace(go.Scatter(
                    x=result_df['Date'],
                    y=result_df[d],
                    mode='lines+markers',
                    name=f"{d} %",
                    yaxis="y2"
                ))
            
        title_text = "SPI+AI Defect Distribution (J15)"
        annotation_text = ""

        # Update layout
        fig.update_layout(
            barmode='stack',
            xaxis_title="Date",
            yaxis=dict(
                title="Job Row Counts",
                side='left'
            ),
            yaxis2=dict(
                title="False Alarm Rate %",
                range=[0, 100],
                overlaying='y',
                side='right',
                anchor='x'
            ),
            legend=dict(x=1.1, y=1)
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            y=0.9,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        template="seaborn",
        hovermode="x unified"
    )

    # --- Add Average Box (Custom Legend) ---
    if annotation_text:
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=1, y=1,
            xanchor="right", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            font=dict(size=12, color="black")
        )

    # Save as PNG
    output_filename = "SPI_AI_Stats.png" # Generic name
    try:
        fig.write_image(output_filename)
        print(f"Chart saved successfully to {output_filename}")
    except Exception as e:
        print(f"Could not save PNG (is 'kaleido' installed?): {e}")

    # Show interactive chart
    fig.show()

def process_consolidated_log(file_path, start_cutoff_dt, mode="default", target_defects=None):
    """
    Processes the consolidated log.csv format.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Consolidated log {file_path} not found.")
        return pd.DataFrame()

    print(f"Processing consolidated log: {file_path}")
    try:
        # Using low_memory=False to avoid DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)
        if 'logged_at' not in df.columns:
            print("Error: 'logged_at' column not found in log.csv")
            return pd.DataFrame()

        # Clean column names
        df.columns = df.columns.str.strip()

        # Handle ISO 8601 with timezone: convert to datetime and strip timezone for comparison
        df['logged_at_dt'] = pd.to_datetime(df['logged_at'], errors='coerce').dt.tz_localize(None)
        df = df.dropna(subset=['logged_at_dt'])
        
        # Ensure start_cutoff_dt is naive for comparison
        cutoff = start_cutoff_dt.replace(tzinfo=None)
        df = df[df['logged_at_dt'] >= cutoff]
        
        if df.empty:
            print("No data found in log.csv after cutoff date.")
            return pd.DataFrame()

        df['Date'] = df['logged_at_dt'].dt.strftime("%Y/%m/%d")
        
        if mode == "default":
            daily = df.groupby('Date').agg({
                'pass_count': 'sum',
                'fail_count': 'sum'
            }).reset_index()
            daily.rename(columns={'pass_count': 'Pass Count', 'fail_count': 'Fail Count'}, inplace=True)
            daily['Total'] = daily['Pass Count'] + daily['Fail Count']
            return daily
        
        elif mode == "read_processed":
            # Map img_numbers to Over/Within Thres job row counts
            df['is_over'] = df['img_numbers'] > 60
            
            def aggregate_processed(x):
                # Total here is the sum of pads/images (img_numbers)
                total = x['img_numbers'].sum()
                over_thres = x[x['is_over']]['img_numbers'].sum()
                within_thres = x[~x['is_over']]['img_numbers'].sum()
                
                res = {
                    'Total': total,
                    'Over_Thres': over_thres,
                    'Within_Thres': within_thres,
                    'anomaly': x['anomaly_count'].sum(),
                    'short distance': x['distance_count'].sum(),
                    'me_thres': x['me_thres_count'].sum()
                }
                
                if target_defects:
                    for d in target_defects:
                        if d not in res:
                            res[d] = 0.0
                return pd.Series(res)

            daily = df.groupby('Date').apply(aggregate_processed).reset_index()
            return daily
    except Exception as e:
        print(f"Error processing consolidated log: {e}")
        return pd.DataFrame()

def calculate_and_plot_improvement():
    # --- Configuration ---
    root_directory = r"D:\Dre\JQ_SPI_02_AI_API\log\csv"
    
    # Filter settings
    start_cutoff_str = "2026/1/14 14:30"
    start_cutoff_dt = datetime.strptime(start_cutoff_str, "%Y/%m/%d %H:%M")
    
    # New Transition date for consolidated log
    log_transition_str = "2026/2/6 00:00"
    log_transition_dt = datetime.strptime(log_transition_str, "%Y/%m/%d %H:%M")

    # Constants
    CODE_PASS = 22
    CODE_FAIL = 23
    
    # Target Defects for "read_processed" mode
    # Added 'anomaly' and 'me_thres' to match new log.csv format
    TARGET_DEFECTS = ["low vol", "high vol", "FM/color", "short distance", "high paste", "anomaly", "me_thres"]

    # Dictionary to store aggregated data: 
    # default: { "YYYY-MM-DD": {'pass': 0, 'fail': 0} }
    # read_processed: { "YYYY-MM-DD": {'total': 0, 'low vol': 0, ...} }
    daily_stats = {}

    print(f"Mode: {MODE}")
    print(f"Searching in: {root_directory}")
    print(f"Filtering files after: {start_cutoff_dt}")
    print("-" * 60)

    # --- File Traversal ---
    files_found_count = 0
    
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(".csv"):
                
                # --- Mode-based File Filtering ---
                if MODE == "default":
                    if "_processed" in file:
                        continue
                elif MODE == "read_processed":
                    # If it's a raw file, only include if the _processed version doesn't exist
                    if "_processed" not in file:
                        processed_file = file.replace(".csv", "_processed.csv")
                        if os.path.exists(os.path.join(root, processed_file)):
                            continue

                filename_stem = os.path.splitext(file)[0]
                
                try:
                    # Parse timestamp from filename (Format: YYYYMMDDHHMMSS)
                    # Handle suffix in filename if present (like _processed) by taking first 14 chars
                    timestamp_str = filename_stem[:14]
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                except ValueError:
                    continue

                # Time Constraint Check
                if file_timestamp < start_cutoff_dt:
                    continue

                # Use consolidated log after transition date
                if file_timestamp >= log_transition_dt:
                    continue

                files_found_count += 1
                file_path = os.path.join(root, file)
                
                # --- Read CSV ---
                try:
                    try:
                        df = pd.read_csv(file_path, skipinitialspace=True)
                    except FileNotFoundError:
                        # Fallback to non-processed file if _processed is missing
                        if MODE == "read_processed" and "_processed" in file:
                            file_path = file_path.replace("_processed.csv", ".csv")
                            df = pd.read_csv(file_path, skipinitialspace=True)
                        else:
                            raise

                    df.columns = df.columns.str.strip() # Clean column headers

                    # Aggregate by Date (for the chart X-axis)
                    date_key = file_timestamp.strftime("%Y/%m/%d")

                    if MODE == "default":
                        if 'is_pass' not in df.columns:
                            continue

                        pass_count = len(df[df['is_pass'] == CODE_PASS])
                        fail_count = len(df[df['is_pass'] == CODE_FAIL])

                        if date_key not in daily_stats:
                            daily_stats[date_key] = {'pass': 0, 'fail': 0}
                        
                        daily_stats[date_key]['pass'] += pass_count
                        daily_stats[date_key]['fail'] += fail_count

                    elif MODE == "read_processed":
                        if date_key not in daily_stats:
                             daily_stats[date_key] = {
                                 'total': 0,
                                 'over_thres_job': 0,
                                 'within_thres_job': 0
                             }
                             for d in TARGET_DEFECTS:
                                 daily_stats[date_key][d] = 0
                        
                        row_count = len(df)
                        if row_count > 60:
                            daily_stats[date_key]['over_thres_job'] += row_count
                        else:
                            daily_stats[date_key]['within_thres_job'] += row_count

                        daily_stats[date_key]['total'] += row_count
                        
                        if 'ai_defect_name' in df.columns:
                            defect_series = df['ai_defect_name'].fillna("").astype(str)
                            for d in TARGET_DEFECTS:
                                 # Case insensitive search
                                 count = defect_series.str.contains(d, case=False, regex=False).sum()
                                 daily_stats[date_key][d] += count

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # --- Process Results ---
    print(f"\nProcessed {files_found_count} valid files from log directory.\n")
    
    # Convert daily_stats to DataFrame
    results_list = []
    if MODE == "default":
        for date_str, counts in daily_stats.items():
            results_list.append({
                "Date": date_str,
                "Pass Count": counts['pass'],
                "Fail Count": counts['fail'],
                "Total": counts['pass'] + counts['fail']
            })
    elif MODE == "read_processed":
        for date_str, counts in daily_stats.items():
            row = {
                "Date": date_str,
                "Total": counts['total'],
                "Over_Thres": counts['over_thres_job'],
                "Within_Thres": counts['within_thres_job']
            }
            for d in TARGET_DEFECTS:
                # Store absolute counts first for combining
                row[d] = counts[d]
            results_list.append(row)
    
    result_df_old = pd.DataFrame(results_list)

    # --- Process Consolidated Log ---
    log_csv_path = r"D:\Dre\JQ_SPI_02_AI_API\test\log_data\log2_new.csv"
    result_df_new = process_consolidated_log(log_csv_path, log_transition_dt, MODE, TARGET_DEFECTS)
    # print(result_df_new)
    # --- Combine and Finalize Rates ---
    if not result_df_new.empty:
        result_df = pd.concat([result_df_old, result_df_new], ignore_index=True)
        # Group by Date to merge if there's overlap
        if MODE == "default":
            result_df = result_df.groupby('Date').agg({
                'Pass Count': 'sum',
                'Fail Count': 'sum',
                'Total': 'sum'
            }).reset_index()
        elif MODE == "read_processed":
            agg_dict = {'Total': 'sum', 'Over_Thres': 'sum', 'Within_Thres': 'sum'}
            for d in TARGET_DEFECTS:
                agg_dict[d] = 'sum'
            result_df = result_df.groupby('Date').agg(agg_dict).reset_index()
    else:
        result_df = result_df_old

    if result_df.empty:
        print("No data found matching criteria.")
        return

    # Calculate Rates
    if MODE == "default":
        result_df['Improve Rate %'] = (result_df['Pass Count'] / result_df['Total'] * 100).fillna(0)
    elif MODE == "read_processed":
        for d in TARGET_DEFECTS:
            result_df[d] = (result_df[d] / result_df['Total'] * 100).fillna(0)

    result_df['Date_Obj'] = pd.to_datetime(result_df['Date'], format="%Y/%m/%d")
    result_df = result_df.sort_values(by="Date_Obj")

    # Display Text Data
    print("=== Daily Statistics ===")
    text_df = result_df.copy()
    
    if MODE == "default":
        text_df['Improve Rate %'] = text_df['Improve Rate %'].round(2)
        print(text_df[['Date', 'Pass Count', 'Fail Count', 'Improve Rate %']].to_string(index=False))
    elif MODE == "read_processed":
        cols_to_show = ['Date', 'Within_Thres', 'Over_Thres'] + TARGET_DEFECTS
        for d in TARGET_DEFECTS:
            text_df[d] = text_df[d].round(2)
        print(text_df[cols_to_show].to_string(index=False))

    # --- Plotly Chart ---
    plot_improvement_data(result_df, MODE, TARGET_DEFECTS)

if __name__ == "__main__":
    calculate_and_plot_improvement()
