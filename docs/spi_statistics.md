# SPI 統計

統計只讀取 `backup_output_root/YYYY-MM-DD/job_id/run_id/run.json` 及同行 `ai_result/manifest.json`，不依賴 SQLite、state_root 或 history_root，也不修改輸入或觸發回傳。

```bat
spi_statistics.bat --output reports
spi_statistics.bat --date 2026-11-09 --output reports
```

也可執行 `python -m app.tools.spi_statistics --config config/ai_server.json --output reports`。

- `jobs.csv`：每次 run 一列，包含 job_id、run_id、狀態、階段時間、耗時、實際回傳結果數量、影像備份與 MiNiFi 匯出狀態。
- `deliveries.csv`：每份 CSV 每次回傳嘗試一列，包含起訖時間、耗時、狀態、錯誤、雜湊、來源完成標記與改名時間。
- `summary.csv`：執行次數、成功／失敗／部分成功／不明次數、回傳 PASS／FAIL 數量、平均回傳耗時與待匯出數。

同一份 CSV 取消 `_DONE` 後重新處理，會有獨立 run_id，分開計算。PASS／FAIL 只計算已確認回傳的 CSV。`--date` 依首次發現的台灣日期篩選，與來源資料夾日期可不同。

結果 manifest 依報告同行路徑讀取，整份備份搬移後仍可統計；舊版沒有 run.json 的備份不會混入新版統計。報告內容損壞時工具會報錯，請保留原檔核對。
