from merge_logs import merge_log_files

if __name__ == "__main__":
    merge_log_files(
        'test/log_data/log_analysis_results_test.csv',
        'test/log_data/02_test.csv',
        'test/log_data/log_analysis_report_test.csv'
    )
