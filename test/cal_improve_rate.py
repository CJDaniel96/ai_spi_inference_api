import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def calculate_and_plot_improvement():
    # --- Configuration ---
    root_directory = r"D:\Dre\JQ_SPI_02_AI_API\log\csv"
    
    # Filter settings
    start_cutoff_str = "2026/1/14 14:30"
    start_cutoff_dt = datetime.strptime(start_cutoff_str, "%Y/%m/%d %H:%M")
    
    # Constants
    CODE_PASS = 22
    CODE_FAIL = 23

    # Mode Selection
    # MODE = "default"        # Original: Excludes "_processed", calculates Pass/Fail Improvement Rate
    MODE = "read_processed" # New: Includes only "_processed", calculates Defect Type %
    
    # Target Defects for "read_processed" mode
    TARGET_DEFECTS = ["low vol", "high vol", "FM/color", "short distance", "high paste"]

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
                    if "_processed" not in file:
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

                files_found_count += 1
                file_path = os.path.join(root, file)
                
                # --- Read CSV ---
                try:
                    df = pd.read_csv(file_path, skipinitialspace=True)
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
                        if 'ai_defect_name' not in df.columns:
                            continue
                        
                        if date_key not in daily_stats:
                             daily_stats[date_key] = {'total': 0}
                             for d in TARGET_DEFECTS:
                                 daily_stats[date_key][d] = 0
                        
                        row_count = len(df)
                        daily_stats[date_key]['total'] += row_count
                        
                        defect_series = df['ai_defect_name'].fillna("").astype(str)
                        for d in TARGET_DEFECTS:
                             # Case insensitive search
                             count = defect_series.str.contains(d, case=False, regex=False).sum()
                             daily_stats[date_key][d] += count

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # --- Process Results ---
    print(f"\nProcessed {files_found_count} valid files from log directory.\n")
    
    if not daily_stats:
        print("No data found matching criteria.")
        return

    # Create DataFrame and Sort
    results_list = []
    
    if MODE == "default":
        for date_str, counts in daily_stats.items():
            pass_c = counts['pass']
            fail_c = counts['fail']
            total = pass_c + fail_c
            
            # Calculate Improve Rate
            if total > 0:
                improve_rate = (pass_c / total) * 100
            else:
                improve_rate = 0.0

            results_list.append({
                "Date": date_str,
                "Pass Count": pass_c,
                "Fail Count": fail_c,
                "Total": total,
                "Improve Rate %": improve_rate
            })
    elif MODE == "read_processed":
         for date_str, counts in daily_stats.items():
            total_count = counts['total']
            
            row = {
                "Date": date_str,
                "Total": total_count
            }
            for d in TARGET_DEFECTS:
                if total_count > 0:
                    row[d] = (counts[d] / total_count) * 100
                else:
                    row[d] = 0.0
            results_list.append(row)

    result_df = pd.DataFrame(results_list)
    result_df['Date_Obj'] = pd.to_datetime(result_df['Date'], format="%Y/%m/%d")
    result_df = result_df.sort_values(by="Date_Obj")

    # Display Text Data
    print("=== Daily Statistics ===")
    text_df = result_df.copy()
    
    if MODE == "default":
        text_df['Improve Rate %'] = text_df['Improve Rate %'].round(2)
        print(text_df[['Date', 'Pass Count', 'Fail Count', 'Improve Rate %']].to_string(index=False))
    elif MODE == "read_processed":
        cols_to_show = ['Date'] + TARGET_DEFECTS
        for d in TARGET_DEFECTS:
            text_df[d] = text_df[d].round(2)
        print(text_df[cols_to_show].to_string(index=False))

    # --- Plotly Chart ---
    print("\nGenerating Plotly chart...")

    # We use make_subplots for dual axis if needed, or manual layout. 
    # Manual layout is often simpler for quick dual-axis.
    fig = go.Figure()

    if MODE == "default":
        # Create text labels: Round to 1 decimal place and add '%'
        label_text = result_df['Improve Rate %'].round(1).astype(str) + '%' # Corrected: removed unnecessary escaping

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
        
    elif MODE == "read_processed":
        # --- Lines (Defect Rates) ---
        for d in TARGET_DEFECTS:
            fig.add_trace(go.Scatter(
                x=result_df['Date'],
                y=result_df[d],
                mode='lines+markers',
                name=d
            ))
            
        title_text = "SPI+AI Defect Distribution (J15)"
        annotation_text = ""

        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis=dict(
                title="False Alarm Rate %",
                range=[0, 100]
            ),
            legend=dict(x=1.05, y=1)
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

if __name__ == "__main__":
    calculate_and_plot_improvement()
