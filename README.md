# SPI AI 單一主控

本版本只保留同一台 AIPC 上的原生 OBB 推論流程。啟動一個主程式，由它管理資料接收、原始備份、模型推論、逐筆排程、SPI 回傳重試、背景結果備份與 MiNiFi 匯出。

接收、推論、發布仍是分開的服務模組，但不再透過 SQLite 或交接資料夾認領工作。模型與可能阻塞的 I/O 放在由主控管理的子程序；SQLite 只保存本機事件的背景查詢副本。

## 啟動

主要程式位置：

```text
app/main.py                         # 單一 CLI 入口
app/application/controller.py       # FIFO 工作生命週期、結果選定及重傳
app/application/ingest/              # CSV／影像驗證與不可變備份
app/application/inference/          # OBB 結果與 CSV 逐列判定
app/application/operations.py        # 受監督的發布與背景備份操作
app/infrastructure/local_store.py   # 本機事件、快照及單一主控鎖
app/infrastructure/audit.py         # 背景 SQLite 投影
app/minifi.py                       # 明確結果 manifest 的 MiNiFi 匯出
```

需要 Python 3.12。Windows / NVIDIA AIPC：

```bat
setup.bat
start_spi_ai.bat
```

`setup.bat` 使用 uv 安裝 inference 與 cuda extras。也可以先自行準備 Python 3.12 環境，透過 `PYTHON_EXE` 指定解譯器。TensorRT 必須另外安裝與實機 CUDA／驅動相符的版本。

跨平台指令：

```sh
python -m app.main --check-config
python -m app.main
python -m app.main --config config/ai_server.json --once --date 2026-11-09
```

`--once` 掃描一批工作，依序處理至每筆回傳成功、重試用完或狀態不明，再各嘗試一次背景備份、匯出、清理與 DB 補登。未完成的背景工作留待重啟繼續。FAILED、INPUT_FAILED、RETURN_FAILED、PARTIAL_RETURN、PUBLICATION_UNKNOWN、HISTORY_CONFLICT（及舊紀錄的 EXPIRED）會回傳 exit code 2；啟動、掃描或本機狀態持久化失敗回傳 1。長駐模式持續掃描、逐筆處理並獨立執行背景工作。`--date` 僅用於 `--once`。

## 單一設定檔

正式設定為 `config/ai_server.json`；可用 `AI_CONFIG_PATH` 或 `--config` 指定另一份完整設定。`config/ai_server.example.json` 是使用相對路徑的範本。相對路徑以專案根目錄解析。

| 區塊 | 用途 |
|---|---|
| identity | site、factory、line、station_id，供判定紀錄及 MiNiFi 使用 |
| paths | SPI 輸入、SPI 回傳、不可變原始備份、本機工作狀態與獨立歷史紀錄根目錄 |
| processing | 影像副檔名、名稱 template、穩定等待及輸入就緒等待上限 |
| delivery | 每份 CSV 最多嘗試次數、重試間隔、單次回傳操作上限 |
| runtime | 掃描、IO 子程序啟動、單次 IO、背景操作的監督時間與間隔 |
| inference.obb | 權重、GPU、批次大小、暖機與單筆推論上限、各類別判定門檻 |
| output | SPI 回傳是否保留 job 資料夾；回傳固定只更新原 CSV 的 is_pass |
| database | 背景紀錄開關、SQLite 位置、補登間隔、鎖等待時間 |
| minifi | 背景匯出開關、輸出位置、處理間隔；輸入固定為已完成的結果備份 |
| logging | 系統日誌目錄及檔名 |

設定啟動時載入及驗證一次，再傳給內部子程序；修改後重啟生效。每筆工作保存當時的設定快照。尚未回傳的工作若在設定變更後重啟，會停止機台回傳並標記 `configuration_changed`，不會用新模型或新目的地偷偷續跑。已回傳工作的背景匯出沿用原始設定快照。

舊的 `reliability`、`processing.folder_images_num_threshold`、`runtime.max_active_jobs`、`server`、`model_clients`、`defect_rules`、`pipeline`、scanner 設定及獨立 `db_export.json`／`yolo_area.json` 已移除；未知設定會直接被拒絕。輸入、回傳、備份、狀態、歷史與 MiNiFi 目錄必須分開，不可互相包含。`state_root` 與 SQLite 應放在 AIPC 本機磁碟；SQLite 必須位於這六個資料目錄之外，預設放在獨立的 `records` 目錄。

`paths.history_root` 為必填的獨立歷史位置，例如 `data/history` 或 `D:/AI_SPI/history`，不可設在備份或工作區內。更換此路徑前先停止程式，將既有歷史目錄完整搬到新位置；由舊版升級時，將 `backup_output_root/_history` 的內容移至新的 history_root，保留 `YYYY-MM-DD/job_id` 層級。只改設定而不搬紀錄，可能讓仍在輸入目錄的舊工作被重新接收。歷史 JSON 不必手動修改，統計會從新位置讀取同行的 manifest。

## 每份工作如何處理

輸入是一個 14 碼時間戳資料夾，內含一份或多份 CSV 與對應影像。主控只自動掃描當日資料夾；歷史日期用 `--once --date` 指定。

1. 首次確認有 CSV 即登記為 `QUEUED`，保存本機事件；以首次觀測順序排隊，同時只處理一筆。模型先暖機，排隊與暖機等待不計入該筆處理耗時。
2. 取出工作後記錄 `processing_started_at`；等 CSV／影像穩定，保存不可變 CSV 快照。影像名稱依 template 與原始字串組成，保留 `001` 等前導零；預設 `{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg`。
3. 驗證對應影像，串流複製並核對雜湊，在備份位置驗證可解碼，再允許推論。輸入就緒等待達上限且已有完整 CSV 時，以全 23 保護回傳；CSV 無法完整解析時標記 `INPUT_FAILED`，不捏造結果。
4. 常駐 OBB 子程序依 `batch_size` 批次推論，超過 60 張仍正常推論。先判斷外觀，再依 CSV 每列的 volume、area、height、shift 門檻判定，見 `docs/sx_obb_inference.md`。模型失敗或單次推論逾時時以全 23 保護。
5. 選定並保存唯一結果，先產生完整 SPI 回傳 CSV 與 SHA-256，再逐份回傳。保留原檔名、欄位、編碼、列序及非 is_pass 值。重試只複製同一份已保存的結果，不重新推論。
6. 全部 CSV 均達到成功、失敗或待確認狀態後，主流程立即處理下一筆。背景另外保存原始資料、回傳版、processed CSV、manifest 與 delivery.json。只有全部 CSV 確認回傳成功的工作才進入 MiNiFi 匯出。

候選清單只是某一時間點的快照。主控在接收前透過受監督的 I/O 再次確認完成紀錄；登記時另同步確認記憶體中的工作、清理中工作與已結束 ID。清理在釋放工作前先保留當日已結束 ID，因此即使舊清單仍含該資料夾、或檔案檢查在清理完成後才回覆／逾時，也不會重新登記同一筆工作。完成紀錄檢查逾時時延後探索，不據此建立新工作。

## 回傳重試與操作監督

AI 不再設定整筆工作的六秒期限，也不因排隊或總耗時變長而強制全 23。SPI 是否接受晚到結果由機台的設定決定。回傳檔案寫入成功只代表輸出位置已確認寫入，沒有 SPI 業務層的接收／接受 ACK。

```json
"delivery": {
  "max_attempts": 3,
  "retry_interval_seconds": 1,
  "attempt_timeout_seconds": 10
}
```

`max_attempts` 包含第一次傳送；3 代表最多嘗試 3 次。計數與每次起訖時間、單調時鐘耗時、狀態、錯誤均寫入本機事件，跨重啟保留。已確認成功的 CSV 不重送；失敗後繼續其他 CSV，整筆結束後處理下一筆。

| 結果 | 處理方式 |
|---|---|
| 確認完成原子改名 | `RETURNED`，記錄回傳時間及從開始處理到回傳的耗時 |
| 確定未送達，例如權限拒絕或改名前已逾時 | 重試同一份結果，達次數上限後該 CSV 記為 `FAILED` |
| 操作逾時、連線中斷或改名確認遺失，可能已送達 | 該 CSV 記為 `UNKNOWN`，不自動重送；繼續其他 CSV／工作 |

整筆工作全成功為 `PRIMARY_RETURNED`，結果備份完成後為 `DONE`；全部明確失敗為 `RETURN_FAILED`，部分成功且其餘失敗為 `PARTIAL_RETURN`，任何 CSV 不明為 `PUBLICATION_UNKNOWN`。失敗與部分成功保留完整結果與紀錄後歸檔，不以成功工作送入 MiNiFi；狀態不明則保留工作區等待人工釐清。

仍保留以下**單一操作**上限以防程式永久卡住，均可在設定檔調整：

| 設定 | 預設秒數 | 範圍 |
|---|---:|---|
| processing.input_ready_timeout_seconds | 30 | 實際開始處理後，等待這筆 CSV／影像穩定及驗證 |
| runtime.io_startup_timeout_seconds | 30 | IO 子程序啟動與載入依賴；掃描 worker 會重用 |
| runtime.discovery_timeout_seconds | 5 | 單次候選掃描／來源檢查／完成紀錄查詢 |
| runtime.io_timeout_seconds | 30 | 單次原始備份、讀取、判定結果寫檔及回傳檔準備 |
| inference.obb.startup_timeout_seconds | 120 | 模型啟動與暖機 |
| inference.obb.inference_timeout_seconds | 120 | 單筆模型推論，涵蓋所有批次 |
| delivery.attempt_timeout_seconds | 10 | 單份 CSV 的單次 SPI 回傳 |
| runtime.background_timeout_seconds | 30 | 單次背景備份、匯出、歸檔或 DB 補登 |

這些時間不是 SPI 有效期限；排隊沒有期限。輸入就緒與單次操作上限在新的執行／嘗試重新計算，回傳次數則不重設。掃描逾時會延後重掃，不據此建立或宣告任務過期。IO 與模型子程序逾時會終止該操作，避免晚到寫入改變結果。

沒有機台完成旗標時，檔案穩定只是完成推估，需確認機台寫檔與來源保留方式。Windows／SMB 伺服器仍可能完成先前送出的 rename，即使本機子程序已被終止，因此不明結果需要依 SPI 記錄人工核對。

系統日誌會顯示 `discovery.ready` 的實際 watch_root 與掃描上限，`discovery.scan` 顯示候選與新排隊數（閒置時約每 30 秒一筆），`delivery.attempt` 顯示檔名、嘗試次數、結果、耗時與錯誤。看到 `obb.ready` 代表模型暖機完成，掃描與佇列狀態可從上述事件確認。

## 本機恢復與 DB

```text
state_root/
  controller.lock
  jobs/<job_id>/              # 尚未完成收尾或回傳狀態待釐清的工作
    events/00000001.json ...   # 不可變事件；含完整工作狀態
    job_state.json            # 可讀快照，可由事件重建
    csv/<generation>/...      # 不可變原始 CSV 快照
  artifacts/<job_id>/<attempt>/
    manifest.json
    processed/...
    returned/...              # 固定回傳內容，重試不重新產生

history_root/
  YYYY-MM-DD/<job_id>/
    receipt.json             # 完成紀錄，供重啟去重
    events/00000001.json ...  # 歷史事件與待補登 DB 的資料
    manifest.json            # 若有選定結果，保存供統計使用的 manifest
backup_output_root/
  YYYY-MM-DD/<job_id>/...     # 原始資料及 ai_result 結果備份

records/audit.sqlite3        # 可重建的背景查詢副本；位於 state_root 外
```

歷史目錄只保存 JSON 紀錄：`receipt.json` 包含最終狀態、處理起訖與耗時、各 CSV 回傳確認及逐次嘗試、備份／匯出結果及設定快照；`events` 保留完整狀態歷程，供 DB 補登與稽核；有選定結果時另保存 `manifest.json`，包含逐 CSV／逐列結果代碼、數量、原因與模型資訊。原始影像、原始 CSV、完整 processed CSV 與 returned CSV 本體保存在 backup_output_root。

只有一個主控能持有同一個 state_root 的 OS 鎖。每次狀態變更先寫入並同步本機事件，再更新快照。重啟由未完成工作的事件恢復，並依 `paths.history_root` 中的完成紀錄略過已結束的來源工作，不讀 SQLite 來決定工作是否完成。不要手動刪除或編輯事件、發布紀錄與 `paths.history_root`；歷史目錄由設定獨立指定，與備份位置分開。

正常工作在全部 SPI CSV 確認回傳、結果備份完成、MiNiFi 匯出完成或停用，以及原始影像備份完成或重試後記錄不完整時，才可清理。主控另確認該筆主流程及背景寫檔已結束；檔案操作均由可終止的子程序執行。先把完整事件、最終摘要與選定結果 manifest 原子保存到 `paths.history_root`，才刪除 `jobs/<job_id>` 及 `artifacts/<job_id>`，並釋放主控記憶體中的該筆工作。清理週期沿用 `runtime.background_retry_seconds`，不阻擋下一筆 SPI 處理。

新產生的 INPUT_FAILED、FAILED、RETURN_FAILED、PARTIAL_RETURN 工作也先保存原始資料與失敗證據，再歸檔清理；舊 EXPIRED 完成紀錄繼續用於去重。PUBLICATION_UNKNOWN 保留在工作區等待釐清，不自動補發或刪除。歷史寫入失敗時保留工作區並重試；清理中途當機時，已提交的完成紀錄仍阻止重送，重啟後繼續刪除殘留資料。`controller.lock` 及空的 jobs／artifacts 根目錄持續保留。

若舊版本已讓同一 job_id 產生不同執行歷程，主控與報表會標示 `HISTORY_CONFLICT`。原始工作事件、快照、結果與歷史紀錄全部保留，該筆工作停止自動推論、回傳、背景作業及清理；每次啟動只報告一次衝突，不阻擋其他工作清理。需核對兩份紀錄與 SPI 接收確認後再人工處理，不能直接刪除歷史紀錄來解除去重。歷史相符、只是中斷刪除的工作仍會自動繼續清理。

每份 SPI CSV 發布前先保存 `PUBLISHING` 意圖，收到成功確認後保存 `RETURNED`。若當機留下未確認的發布意圖，重啟標記狀態不明，不根據機台目錄是否還有檔案判斷成功，也不盲目補發。已確認全部 CSV 回傳、但尚未保存整筆完成狀態的工作，能繼續結果備份，不受停機時間長短影響。其餘未送出的 CSV 仍依保存的計數繼續；未確認的那一份不補發。

背景 DB 程序讀取工作區與 `paths.history_root` 事件，以唯一 event_id 與 job_id/revision 冪等寫入 `events`、`jobs` 兩張表。DB 鎖定、離線或程序被終止時，事件保存在工作區或歷史目錄，之後補登；DB 補登成功不是清理的前提。關閉 `database.enabled` 不影響接收、推論、SPI 回傳、清理或恢復。DB 不保存影像或 CSV 本體。既有 audit.sqlite3 的 jobs 表會在背景交易中改為新欄位；原始事件與歷史 snapshot JSON 保留，不用刪除 DB。既有部署若把 SQLite 放在 state_root，更新時需停止舊程式、將 `database.path` 改到外部目錄；新 DB 可由保留的事件重建，舊 DB 檔案請一併移出工作區。

發現歷史衝突時，DB 的 `history_conflicts` 表另外保留歷史與工作區兩份狀態，jobs 標示 HISTORY_CONFLICT，重複執行的事件不混入原歷史序號。已存在的錯誤補登資料不會自動刪除，需與原始紀錄一併核對；DB 仍只做背景紀錄。系統日誌的狀態事件包含 `last_error`，可直接追查 CSV 輸入與模型失敗原因；模型暖機超時標記 `required_model_timeout`。

主控本身故障時接收與回傳會停止；應由現場的服務管理機制重新啟動。重啟不重設回傳次數，也不以停機多久判斷工作過期。升級前已完成的歷史紀錄會繼續用於去重；舊設定快照的未完成工作會依 configuration_changed 停止自動回傳，應先核對這些工作。

## 備份、MiNiFi 與統計

原始備份位置為 `backup_output_root/YYYY-MM-DD/job_id/`。結果保存在同一工作下的 `ai_result/returned`、`ai_result/processed`、`manifest.json` 與 `delivery.json`；匯出成功的 processed CSV 移到 `ai_result/exported`。`DONE` 表示 SPI 回傳與本機結果備份完成，MiNiFi 匯出可能仍在等待。

MiNiFi 只收集輸出目錄中的 `*.csv`，忽略 `.tmp`。背景程序依明確選定的結果 manifest 匯出，不掃描未選定的推論嘗試。NG 中無缺陷名稱的全 23 保護列標記 `AI_SKIP`。輸出完成不代表 MiNiFi 已上傳；上傳與下游資料庫由 MiNiFi 負責。部分輸出失敗後重試可能重複已被收走的 CSV，下游須依穩定 UUID 去重／upsert。

```bat
spi_statistics.bat --output reports
spi_statistics.bat --date 2026-11-09 --output reports
```

統計合併工作區與 `paths.history_root` 的不可變事件，同一 job/revision 只計算一次；結果 manifest 使用仍存在的工作檔或歷史副本。不依賴 DB 補登進度，也不修改工作狀態。輸出 jobs.csv、逐次回傳的 deliveries.csv 與 summary.csv；日期以 AI 首次觀測的台灣日期篩選。

## 模型格式與離線匯出

OBB 支援 `.pt`、`.onnx`、`.engine`。模型必須是七類 SPI OBB；不支援舊 PatchCore、distance 或 segmentation HTTP 串接。`.engine` 需要 NVIDIA CUDA；`engine_fixed_batch` 只對 `.engine` 生效，固定批次尾批會補圖並丟棄補圖結果。ONNX 固定批次從模型 shape 讀取。

可用同一套 Ultralytics 後端離線匯出 `.pt`：

```bat
convert_to_tensorrt.bat --model models/appearance/best.pt --output models/appearance/best.engine --device cuda:0 --batch 8 --half
```

輸出副檔名也可為 `.onnx`。預設固定批次 8；若使用 `--dynamic` 匯出 engine，推論設定的 `engine_fixed_batch` 要改為 false。TensorRT engine 應在實際 AIPC 的 GPU／CUDA 環境產生並驗證。匯出工具只接受 OBB 七類權重，先在暫存目錄完成匯出，再發布輸出檔。

## 開發與驗證

```sh
uv sync --group dev
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

不需要 GPU 就能執行設定、CSV、判定、逐筆處理、回傳重試、DB lock、恢復與子程序監督測試。真實程序整合測試使用假模型；不代表實際權重精度或 Windows／SMB 現場行為已驗證。要在 macOS 執行真實 CPU 模型可安裝 `--extra inference --extra mac`。

本分支不保留舊 API／scanner／三 Worker 相容入口，也不遷移舊 pipeline.sqlite3。切換前停止舊程式、保留舊資料，並確認輸入目錄只包含要由新版處理的工作；不要同時啟動兩種架構。
