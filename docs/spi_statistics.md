# 單一主控統計

執行 `spi_statistics.bat --output reports`，或 `python -m app.tools.spi_statistics --output reports`。可指定同一份 `--config` 及 `--date YYYY-MM-DD`。

工具合併 `state_root/jobs` 與 `paths.history_root` 的不可變事件，同一 job/revision 只計算一次。已完成工作的 manifest 讀取歷史副本，執行中工作讀取其選定結果。不讀取或修改 SQLite，也不覆寫 job_state.json。日期依首次觀測的台灣時間篩選。

| 檔案 | 內容 |
|---|---|
| jobs.csv | 每筆工作的狀態、輸入／處理／回傳時間、耗時、正常或 fallback、影像／判定數、回傳成功／失敗／不明份數、備份與匯出狀態 |
| deliveries.csv | 每份 CSV 每次嘗試的序號、起訖、耗時、結果、錯誤、最終回傳狀態與目的地、來源與結果 SHA-256 |
| summary.csv | 完整回傳、失敗、部分回傳、不明、歷史衝突工作數，以及已完整回傳的品質數量與平均耗時 |

時間欄位：

- `first_seen_at`：AI 首次觀測到 CSV，並非機台寫檔完成時間。
- `processing_started_at`：暖機完成、從 FIFO 佇列取出開始處理。重啟保留第一次開始時間。
- `queue_ms`：首次觀測到第一次開始處理，包含排隊及暖機等待。
- `processing_finished_at`：所有 CSV 的回傳狀態已確定或記為不明；不包含後續備份／匯出。
- `processing_elapsed_ms`：已保存的主流程執行耗時，以單調時鐘累加；排隊及停機不累加。突然斷電後，最後一個事件之後尚未保存的那一小段無法精確重建。
- `processing_to_return_ms`：第一次開始處理至檔案回傳確認的牆鐘耗時，跨重啟時包含停機時間；jobs.csv 以最後一份確認為準，deliveries.csv 保留各 CSV 的耗時。不能當成 SPI 接受結果的 ACK。
- `elapsed_ms`：單次傳送嘗試的單調時鐘耗時；當機留下的未確認嘗試沒有可靠結束耗時，保留空白，不推估成成功。

不再輸出整筆期限、deadline_met 或到期達成率。舊歷史紀錄保留原始 JSON；缺少新版時間欄位時，報表留白或依可用資料呈現，不推估成新制耗時。deliveries.csv 只列出實際保存的逐次嘗試，舊紀錄若沒有 attempt_log 不捏造明細。

summary.csv 的品質列數只計算已完整回傳 SPI 的工作。PARTIAL_RETURN 保留部分成功數；PUBLICATION_UNKNOWN 不當成確定成功或失敗。失敗或不明工作不作成功的 MiNiFi 匯出。檔案輸出完成也不代表 MiNiFi 已上傳。

若同一 job_id 在工作區與歷史出現不同事件 ID 或序號，報表標示 HISTORY_CONFLICT，summary.csv 的 history_conflict_jobs 計算待核對數。兩份原始資料均保留，重複執行不合併到原歷史序號，也不列入回傳成功與品質統計。
