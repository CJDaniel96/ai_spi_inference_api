# 三階段程式結構

三個正式流程階段各自集中在 `app/application/` 下，以職責命名資料夾。
根目錄的 `01_pipeline_ingest.bat`、`02_pipeline_inference.bat`、`03_pipeline_publisher.bat`
仍透過 `python -m app.pipeline <stage>` 啟動；CLI 負責參數解析、路徑處理、設定、啟動檢查與資源組裝。

```text
app/
  pipeline.py
  application/
    ingest/
      worker.py             # 01：掃描、穩定觀察、認領、期限及狀態推進
      service.py            # CSV／影像驗證、不可變原始備份
      io.py                 # 可供 spawned 子程序執行的接收與原始影像備份函式
    inference/
      worker.py             # 02：依期限認領工作、保存結果狀態、排除晚到結果
      contracts.py          # 共用的結果準備介面
      obb_service.py        # SX：常駐 OBB、CSV 逐列門檻判定與結果保存
      service.py            # Legacy HTTP：模型呼叫、合併與舊規則判定
    publisher/
      worker.py             # 03：截止接管、Primary 回傳、備份與匯出通道排程
      service.py            # 機台 CSV 發布及本機結果備份
      backup.py             # 可供 spawned 子程序執行的結果備份函式
      db_export_worker.py   # 背景監督 MiNiFi OK／NG CSV 匯出
    shared/
      fallback_result.py    # 三個階段共用的全 23 結果產生器
      result_artifacts.py   # 正常／保護性結果共用的 processed CSV 寫入與目錄配置
    use_cases/              # 相容 API 的同步處理流程
  domain/                   # 工作狀態、manifest、缺陷規則、衍生數值
  infrastructure/           # SQLite、模型 HTTP client、通用 I/O 與機台 CSV adapter
    inference/              # OBB 後端與常駐模型子程序；GPU 函式庫只在子程序載入
  core/                     # 設定、日誌、例外
```

每個階段的 `worker.py` 處理工作認領、租約、重試與狀態轉換，`service.py` 執行該階段的資料處理。
02 依 `inference.mode` 選擇 native `obb_service.py` 或 legacy `service.py`，同用結果準備介面。
接收與備份的專用子程序函式放回所屬階段，讓共用 infrastructure 元件只處理通用機制。

三個階段都使用 `shared.fallback_result.FallbackResultBuilder`：

- 01：影像超量時，從不可變 CSV 快照產生全 23。
- 02：必要模型失敗、逾時或資料不完整時，產生全 23 並保留模型耗時與錯誤。
- 03：到推論截止點時，從有效的 CSV 快照或本機輸入接管，產生全 23。

CSV-only 路徑不掃描或解碼影像。01、03 與共用 fallback 不載入 02 的模型呼叫服務。
CLI 在選定階段後才載入相關模組；Publisher 與共用 fallback 不需要接收階段的 OpenCV runtime。
正常判定與 fallback 透過 `ResultArtifactWriter` 使用相同的嘗試目錄、processed CSV 檔名及逐列結果格式。

設定檔、SQLite schema、工作狀態、原始六秒期限、輸出路徑及 CLI 參數沿用既有契約。
`db_export.bat`、`spi_statistics.bat` 仍是獨立工具，03 的背景 MiNiFi 匯出照常運作。
更新部署時先停止三個 Worker，完整替換程式檔案後重新啟動，讓所有 spawned 子程序使用新的模組路徑。
