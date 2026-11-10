# 單一主控統計

執行 `spi_statistics.bat --output reports`，或 `python -m app.tools.spi_statistics --output reports`。可指定同一份 `--config` 及 `--date YYYY-MM-DD`。

工具合併 state_root/jobs 與 paths.history_root 下的不可變事件，同一 job/revision 只計算一次。已完成工作的 manifest 讀取歷史副本，尚在執行的工作讀取其選定結果。不讀取或修改 SQLite，也不重建／覆寫 job_state.json。工作區清理及 DB 尚未補登都不會使已保存的歷史統計消失。日期依首次觀測的台灣時間篩選。

jobs.csv 包含工作狀態、原因、首次觀測來源、期限、SPI 回傳時間與耗時、正常／fallback、影像數、判定列數、模型耗時、逐 CSV 回傳數、原始影像備份、結果備份及 MiNiFi 匯出狀態。

summary.csv 的品質列數只計算已完整回傳 SPI 的工作。部分回傳工作仍保留 returned_csv_count 與終止原因；PUBLICATION_UNKNOWN 不當成確定成功或失敗。檔案輸出成功不代表 MiNiFi 已上傳。

first_seen_source=folder_inspection 表示首次目錄檢查卡住時的保守觀測，不能當成已確認 CSV 寫入完成。AI 首次觀測也不是機台送出完成事件。
