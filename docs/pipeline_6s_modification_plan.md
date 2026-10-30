# SPI 完整流程期限與超量快速回寫修改計劃

日期：2026-10-30。狀態：已實作並完成本機回歸驗證，尚未部署產線。

目前設定為整體 6 秒、發布保留 1 秒。涵蓋 CSV-only 快速回寫、絕對期限、逐檔發布紀錄、有限重試、逾時終止、受監督的 I/O 子行程、背景影像備份及新版統計。Windows／SMB 現場耗時及機台來源保留行為仍須部署前驗證。首次目錄檢查本身卡住時，採保守的目錄觀測期限並在報表標記 `folder_inspection`，不冒充已觀測到 CSV。

## 目標與範圍

本次調整資料接收、超量分流、CSV 發布、期限控制與統計。超過設定影像數門檻的工作，在確認 CSV 完整後建立全 23 結果並原子發布，不等待影像解碼或完整影像備份。回寫失敗可以重試，但不得重新取得一份完整時間預算；整體期限到達後，停止該工作的機台回傳並處理其他工作。

模型重新訓練與改為單一 YOLO segmentation 由後續工作處理。本次不改 distance 空值的品質判定規則、不加速 anomaly，也不導入新模型。正常推論仍透過既有通用模型 client 呼叫，遵守新的整體剩餘時間。

## 1. 可設定的完整流程期限

新增 `reliability.end_to_end_timeout_seconds`，預設 6.0 秒，必須大於 0。Pipeline 使用此值計算唯一的絕對截止時間；原 `primary_return_deadline_seconds` 保留給舊 `/process` 路徑，不讓兩份預算先後相加。

正式及範例設定一併調整：

```json
{
  "reliability": {
    "end_to_end_timeout_seconds": 6.0,
    "primary_publish_reserve_seconds": 1.0
  }
}
```

這是局部設定範例。`primary_publish_reserve_seconds` 沿用現有欄位，1.0 秒為初始發布預算，需用現場測量校準，且必須小於完整流程期限。

理想計時起點是機台的本次資料寫入完成事件。目前介面沒有可靠的完成事件，本版計劃保守地從「首次偵測到本次 CSV」開始計時，將 CSV 穩定等待、完整性確認、影像處理、排隊與發布重試全部計入。此欄位不能命名或解釋成機台送出完成時間；若日後取得完成旗標，才能另行精確比較他站的同口徑時間。

首次觀測與 deadline 寫入本機 SQLite。同 job 重複掃描、工作重試、Worker 重啟都不重設起點。各階段使用剩餘時間；進程內等待使用單調時鐘，跨進程及重啟使用持久化的 UTC 截止時間。

## 2. CSV 完整性與影像完整性分開

建立獨立的 CSV readiness 檢查，不再要求先把每張影像讀取、解碼後才能分流。

- 確認 CSV 檔案集合、大小與修改時間在穩定窗口內一致。初版沿用現有穩定等待值，不直接任意縮短。
- 驗證 CSV 可以完整解析、必要欄位及 `is_pass` 存在、資料列和影像名稱對應合法。
- 在讀取前後核對來源快照，將原始 CSV 保存為本機不可變快照，確保回傳列序與內容一致。
- 一個時間戳資料夾有多份 CSV 時，穩定判斷涵蓋整個 CSV 集合，不能只處理第一份。
- 沒有完成旗標時，檔案穩定只是可操作的完成推估，不能證明機台不會再追加另一份 CSV。部署時需核對機台寫檔契約及最大檔案間隔。
- CSV 尚未完整就耗盡預算時，記錄逾時並停止處理，不產生來源不完整的全 23 檔案。

正常工作仍檢查所需影像的完整性，然後走現有本機備份與推論流程。

## 3. 超過 60 張的快速路徑

門檻沿用 `processing.folder_images_num_threshold`，保持嚴格大於：60 張仍推論，61 張開始跳過。

```text
發現 CSV，持久化開始時間與截止時間
  → CSV 完整性檢查及本機 CSV 快照
  → 按設定的影像副檔名列舉實際影像
      → 超過門檻：建立全 23 manifest，直接排入發布
      → 未超過門檻：等待影像完整、本機備份、既有推論、發布
```

超量判斷使用實際影像檔案數，不擅自把 CSV 列數當影像數。超過門檻即可停止計數，不解碼影像、不呼叫任何模型。

影像可能在 CSV 穩定後持續送入。第一次計數未超量時，等待影像的期間必須持續檢查門檻；一旦實際影像數達到 61 張，就切換快速路徑，不等整批影像到齊才分流，且不重設原 deadline。

新增接受 CSV 快照的 fallback builder，沿用既有 CSV 編碼、欄位順序、列序、原檔名與原子發布介面。快路徑不得經由會重新載入、驗證整批影像的 helper。

Publisher 對快路徑要求「CSV 快照及結果已持久化」，不再要求完整原始影像備份旗標。正常推論路徑保留既有原始資料要求；不得用只有 CSV 的備份冒充 `raw_backup_ready`。

原始影像備份改為獨立的低優先工作，不阻塞快路徑或下一筆主工作。記錄 pending/complete/incomplete/failed 等備份狀態，與是否已回傳分開。若機台回傳後立即刪除來源，背景影像備份可能不完整；要保證影像留存，需機台保留期或另一個可靠影像來源。本次快速路徑以準時回傳為優先，至少完整保存原始 CSV、結果及發布紀錄。

## 4. 重試只使用整體剩餘時間

- 每次 claim、每次發布及每次重試前，都檢查持久化的絕對 deadline。
- 未到期且寫入失敗時，保留錯誤與嘗試次數，設定短暫 `next_retry_at`。重試間隔不得超出剩餘預算，等待期間可處理其他工作。
- 到期的未完整發布工作轉為終止狀態 `EXPIRED`，記錄階段、原因、最後錯誤及時間。發布成功後的本機結果備份不受機台回傳期限中止。
- 到期後不自動重新 claim、不重新掃描排入、不補發、不因重啟重新計時。已在執行的推論或本機備份即使較晚完成，也不能恢復機台發布資格。
- 多 CSV 逐檔記錄發布結果，避免重試重送已成功的檔案。到期時保留已完成的發布，不再發布剩餘檔案，明確標記部分發布。
- `DONE`、`FAILED` 與 `EXPIRED` 具有不同語義，不能把尚未回傳的工作記成成功。

## 5. 阻塞 I/O 與晚到發布

只在重試迴圈檢查 deadline，無法處理單次共享磁碟呼叫卡住。已執行的 thread 工作也不能靠 `Future.cancel()` 停止。

計劃把可能阻塞的共享資料夾讀取／發布放在可監督的獨立 I/O 子行程；資料庫狀態與 lease 由父行程管理，子行程不持有共用的 SQLite transaction。正常情況重用已啟動的 I/O 行程；超時則隔離、終止及回收，限制清理等待，避免主 Worker 被拖住。

寫入仍在目的目錄產生暫存檔，完成後進行原子 rename。最後發布前再次確認 deadline 與發布資格。暫存檔清理由後續低優先工作處理，不阻塞下一筆。

必須區分兩種保證：系統可以在期限到後停止重試及排程舊工作，但已送往 SMB 伺服器的 rename 可能在回應中斷後仍完成，不能承諾跨網路的絕對撤銷。無法確認最後發布是否完成時，記錄 `publish_completion_unknown`，不記成確定未發布，也不自動重送。若要求機台絕不接受晚到結果，需機台一起檢查 job ID 與有效期限。

正常模型服務的 GPU 工作未必能因取消 HTTP 等待而停止；本次保證主流程丟棄晚到結果及停止該工作發布，不把它當成模型服務已具備運算搶占。

## 6. SQLite 與統計

擴充狀態以支援 CSV 尚未完成時就持久化觀測與期限，以及 CSV-only 的超量 fallback。具體欄位包含：首次偵測時間、CSV ready 時間、期限起點與來源、CSV 快照位置／完整性、快路徑類型、下次重試時間、到期時間、逐 CSV 發布紀錄及獨立影像備份狀態。

既有 `ready_at` 保留「CSV／影像完整驗證通過」的定義，不把新起點塞入同名欄位。快路徑沒有完整影像驗證時允許該值缺漏，統計改用各自有意義的時間欄位。

遷移需處理目前 `ready_at NOT NULL`、deadline check 及固定 status check；不是只加 Python enum。先備份資料庫，停妥三個 Worker 後執行版本化遷移。保留歷史紀錄與既有工作的期限，不追溯重算或重置；舊版 Worker 不混用新 schema。

統計工具新增：

- `end_to_end_s`、起點來源及 P50/P95/P99/max。
- 依設定期限計算達標、逾時放棄、部分發布與完成狀態未知的數量。
- 快路徑數量、發布重試次數及影像備份完整性。
- 逾時放棄的工作沒有成功延遲值，但必須留在結果分母中，不能只算成功回傳的工作而高估達標率。
- 保留舊的 `internal_total_s` 意義及歷史報表可讀性。

## 7. 主要修改位置

| 範圍 | 主要檔案 |
|---|---|
| 設定與驗證 | `app/core/config.py`、`config/ai_server.json`、`config/ai_server.example.json` |
| CSV readiness 與觀測 | `app/infrastructure/input/sinic_folder_input.py`、`app/application/services/ingest_service.py`、`app/application/workers/ingest_worker.py` |
| 狀態、遷移與期限 | `app/domain/entities/pipeline_job.py`、`app/infrastructure/repositories/pipeline_job_repository.py` |
| 超量結果與晚到防護 | `app/application/services/pipeline_inference_service.py`、`app/application/workers/inference_worker.py` |
| 發布、重試與 I/O 監督 | `app/application/services/pipeline_publisher_service.py`、`app/application/workers/publisher_worker.py`、`app/infrastructure/output/sinic_csv_output.py`、新增 I/O 監督模組 |
| Worker 啟動與資源回收 | `app/pipeline.py` |
| 報表、文件及回歸測試 | `app/tools/spi_statistics.py`、`docs/spi_statistics.md`、`README.md`、相關 `tests/unit/` 與 `tests/integration/` |

## 8. 實作順序與驗收

依相依關係分四批：

1. 設定、期限資料模型、SQLite 遷移、CSV readiness。
2. CSV-only 超量結果、快速發布與獨立影像備份。
3. 到期終止、有限重試、I/O 子行程監督與晚到結果防護。
4. 統計、文件、整合回歸及 Windows／SMB 現場驗證。

關鍵驗收案例：

| 情境 | 預期結果 |
|---|---|
| 60 張 | 正常推論路徑，仍受同一整體期限約束 |
| 61 張及 2,115 張 | 不解碼影像、不呼叫模型、不等待完整影像備份；完整 CSV 全部結果為 23 |
| CSV 正在追加／第二份 CSV 到達 | 未穩定前不發布，起點及期限不重設 |
| CSV 已穩定但影像持續送入，之後達到 61 張 | 等待期間即切換快速路徑，不等待剩餘影像到齊或解碼 |
| 非法 CSV 或缺少必要欄位 | 不回傳錯列或半份結果；逾時終止並記錄原因 |
| 正常推論快到截止 | 使用既有保護性結果機制，但只使用剩餘發布預算 |
| 目的地短暫不可寫，期限內恢復 | 重試並原子發布；不重送已完成的 CSV |
| 目的地持續不可寫 | 到期終止舊工作，下一筆仍可排程 |
| 單次共享磁碟 I/O 卡住 | 父行程不無限等待；舊任務隔離回收，未知提交結果另記 |
| 多 CSV 只完成部分回寫後到期 | 保留逐檔紀錄，停止剩餘發布，標記部分發布逾時 |
| Worker 重啟或重複掃描 | 沿用既有期限；逾時工作不自動重送 |
| 舊推論在工作到期後完成 | 不恢復狀態、不發布晚到結果 |
| 快路徑回傳後來源影像被刪除 | 主結果維持正確，影像備份明確記為不完整 |
| 歷史 DB 與統計檔 | 遷移保留原紀錄；舊時間意義不變；逾時工作計入報表分母 |

使用注入時鐘驗證期限邊界及重啟行為，使用故障注入驗證寫入失敗與阻塞 I/O；不讓測試依賴真實等待 6 秒。正常原子發布、CSV 編碼／列序、lease 接管及資料庫匯出流程須通過既有回歸測試。實際 6 秒達標率及阻塞取消效果另外在 Windows 產線同型 SMB 環境驗證，不能以本機單元測試替代。

## 技術參考

- [Python 3.12 concurrent.futures：執行中的 Future 無法取消](https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.Future.cancel)
- [Python 3.12 multiprocessing：子行程終止及其資源限制](https://docs.python.org/3.12/library/multiprocessing.html#multiprocessing.Process.terminate)
