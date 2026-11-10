# SPI AI 單一主控

本版本只保留同一台 AIPC 上的原生 OBB 推論流程。啟動一個主程式，由它管理資料接收、原始備份、模型推論、六秒期限、SPI 回傳、背景結果備份與 MiNiFi 匯出。

接收、推論、發布仍是分開的服務模組，但不再透過 SQLite 或交接資料夾認領工作。模型與可能阻塞的 I/O 放在由主控管理的子程序；SQLite 只保存本機事件的背景查詢副本。

## 啟動

主要程式位置：

```text
app/main.py                         # 單一 CLI 入口
app/application/controller.py       # 工作生命週期、期限、結果選定及發布
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

`--once` 掃描一批工作，等候各工作完成回傳或到期，再嘗試一次背景備份、匯出與 DB 補登。失敗的背景工作保留在本機，重啟後繼續。若本次有 FAILED、EXPIRED 或 PUBLICATION_UNKNOWN 工作，--once 會以 exit code 2 結束；背景補登失敗則保留紀錄等待下次處理。長駐模式持續掃描輸入並獨立執行背景工作。`--date` 僅用於 `--once`。

## 單一設定檔

正式設定為 `config/ai_server.json`；可用 `AI_CONFIG_PATH` 或 `--config` 指定另一份完整設定。`config/ai_server.example.json` 是使用相對路徑的範本。相對路徑以專案根目錄解析。

| 區塊 | 用途 |
|---|---|
| identity | site、factory、line、station_id，供判定紀錄及 MiNiFi 使用 |
| paths | SPI 輸入、SPI 回傳、不可變原始備份、本機工作狀態與獨立歷史紀錄根目錄 |
| processing | 影像副檔名、影像名稱 template、實際影像數上限、穩定等待 |
| reliability | 唯一整體期限及發布預留時間 |
| runtime | 掃描與監督間隔、同時接收數量、背景嘗試期限與重試間隔 |
| inference.obb | 權重、GPU、批次大小、模型過濾與各類別判定門檻 |
| output | SPI 回傳是否保留 job 資料夾；回傳固定只更新原 CSV 的 is_pass |
| database | 背景紀錄開關、SQLite 位置、補登間隔、鎖等待時間 |
| minifi | 背景匯出開關、輸出位置、處理間隔；輸入固定為已完成的結果備份 |
| logging | 系統日誌目錄及檔名 |

設定啟動時載入及驗證一次，再傳給內部子程序；修改後重啟生效。每筆工作保存當時的設定快照與期限。尚未回傳的工作若在設定變更後重啟，會停止機台回傳並標記 `configuration_changed`，不會用新模型或新目的地偷偷續跑。已回傳工作的背景匯出沿用原始設定快照。

舊的 `server`、`model_clients`、`defect_rules`、`pipeline`、scanner 設定及獨立 `db_export.json`／`yolo_area.json` 已移除；未知設定會直接被拒絕。輸入、回傳、備份、狀態、歷史與 MiNiFi 目錄必須分開，不可互相包含。`state_root` 與 SQLite 應放在 AIPC 本機磁碟；SQLite 必須位於這六個資料目錄之外，預設放在獨立的 `records` 目錄。

`paths.history_root` 為必填的獨立歷史位置，例如 `data/history` 或 `D:/AI_SPI/history`，不可設在備份或工作區內。更換此路徑前先停止程式，將既有歷史目錄完整搬到新位置；由舊版升級時，將 `backup_output_root/_history` 的內容移至新的 history_root，保留 `YYYY-MM-DD/job_id` 層級。只改設定而不搬紀錄，可能讓仍在輸入目錄的舊工作被重新接收。歷史 JSON 不必手動修改，統計會從新位置讀取同行的 manifest。

## 每份工作如何處理

輸入是一個 14 碼時間戳資料夾，內含一份或多份 CSV 與對應影像。主控只自動掃描當日資料夾；歷史日期用 `--once --date` 指定。

1. 首次觀測 CSV 即保存工作與期限；來源 CSV 集合在穩定窗口內不變後，保存不可變 CSV 快照。
2. 依 template 與 CSV 原始字串組成影像名稱，保留 `001` 等識別值的前導零。預設為 `{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg`。
3. 正常工作驗證影像、串流複製並核對雜湊，在本機驗證影像可解碼，再允許推論。
4. 常駐 OBB 子程序執行推論；同一時間只跑一筆 GPU 工作。先判斷外觀，再依 CSV 每列的 volume、area、height、shift 門檻判定，細節見 `docs/sx_obb_inference.md`。
5. 主控只選定一份結果，保存後依原檔名逐份回傳 SPI。原 CSV 的欄位、編碼、列序與非 is_pass 值保留。
6. SPI 回傳成功後，背景保存回傳版、完整 processed CSV 與 manifest，再轉成 MiNiFi OK／NG CSV。這些背景工作不阻擋下一筆回傳。

超過 60 張實際影像時（嚴格大於；60 張仍推論），保存完整 CSV 後直接全 23，不等待影像解碼與完整原始影像備份。原始影像改為背景嘗試；若 SPI 回傳後來源已刪除，可能無法完整留存，狀態會明確記錄。CSV 列數不作為影像數。

## 六秒期限與保護

```text
first_seen_at       = AI 首次觀測到 CSV 的時間
inference_cutoff_at = first_seen_at + 6 秒 - 1 秒
primary_deadline_at = first_seen_at + 6 秒
```

沒有機台完成旗標時，檔案穩定只是完成推估，部署時需確認機台寫檔與來源保留契約。

六秒涵蓋檔案穩定等待、驗證、正常原始備份、GPU 排隊、推論及 SPI 回傳。結果提早完成就立即發布；到推論截止點仍無有效結果，但已有完整 CSV 快照時，主控選擇全 23。CSV 無法完整解析時不捏造全 23，到期停止。

重試與重啟不重新給時間。程序內另使用單調時鐘，避免系統時鐘往回調整延長同一筆工作。等待發布的工作依截止時間排序，進行中的共享磁碟操作有受監督的期限；晚到模型結果不會覆蓋已選定結果。

本系統為軟即時流程。Windows／SMB、磁碟與 GPU 實際耗時需在現場量測；遠端 SMB 伺服器可能完成已送出的 rename，即使回應逾時或本機終止了 I/O 子程序。無法確認的發布會標記 `PUBLICATION_UNKNOWN` 並停止自動重送。若要機台絕對拒絕晚到結果，需要 SPI 端也驗證工作識別與有效期限。

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

history_root/
  YYYY-MM-DD/<job_id>/
    receipt.json             # 完成紀錄，供重啟去重
    events/00000001.json ...  # 歷史事件與待補登 DB 的資料
    manifest.json            # 若有選定結果，保存供統計使用的 manifest
backup_output_root/
  YYYY-MM-DD/<job_id>/...     # 原始資料及 ai_result 結果備份

records/audit.sqlite3        # 可重建的背景查詢副本；位於 state_root 外
```

歷史目錄只保存 JSON 紀錄：`receipt.json` 包含最終狀態、期限、各 CSV 回傳確認、備份／匯出結果及設定快照；`events` 保留完整狀態歷程，供 DB 補登與稽核；有選定結果時另保存 `manifest.json`，包含逐 CSV／逐列結果代碼、數量、原因與模型資訊。原始影像、原始 CSV、完整 processed CSV 與 returned CSV 本體保存在 backup_output_root。

只有一個主控能持有同一個 state_root 的 OS 鎖。每次狀態變更先寫入並同步本機事件，再更新快照。重啟由未完成工作的事件恢復，並依 `paths.history_root` 中的完成紀錄略過已結束的來源工作，不讀 SQLite 來決定工作是否完成。不要手動刪除或編輯事件、發布紀錄與 `paths.history_root`；歷史目錄由設定獨立指定，與備份位置分開。

正常工作在全部 SPI CSV 確認回傳、結果備份完成、MiNiFi 匯出完成或停用，以及原始影像備份完成或重試後記錄不完整時，才可清理。主控另確認相關子程序與本機寫檔已結束，包含被取消推論留下的背景執行緒。先把完整事件、最終摘要與選定結果 manifest 原子保存到 `paths.history_root`，才刪除 `jobs/<job_id>` 及 `artifacts/<job_id>`，並釋放主控記憶體中的該筆工作。清理週期沿用 `runtime.background_retry_seconds`，不影響 SPI 回傳期限。

FAILED、EXPIRED 工作也在保存最終摘要後清理；PUBLICATION_UNKNOWN 保留在工作區等待釐清，不自動補發或刪除。歷史寫入失敗時保留工作區並重試；清理中途當機時，已提交的完成紀錄仍阻止重送，重啟後繼續刪除殘留資料。`controller.lock` 及空的 jobs／artifacts 根目錄持續保留。

每份 SPI CSV 發布前先保存 `PUBLISHING` 意圖，收到成功確認後保存 `RETURNED`。若當機留下未確認的發布意圖，重啟標記狀態不明，不根據機台目錄是否還有檔案判斷成功，也不盲目補發。已確認全部 CSV 回傳、但尚未保存整筆完成狀態的工作，能繼續結果備份，即使重啟時已超過六秒。

背景 DB 程序讀取工作區與 `paths.history_root` 事件，以唯一 event_id 與 job_id/revision 冪等寫入 `events`、`jobs` 兩張表。DB 鎖定、離線或程序被終止時，事件保存在工作區或歷史目錄，之後補登；DB 補登成功不是清理的前提。關閉 `database.enabled` 不影響接收、推論、SPI 回傳、清理或恢復。DB 不保存影像或 CSV 本體。既有部署若把 SQLite 放在 state_root，更新時需停止舊程式、將 `database.path` 改到外部目錄；新 DB 可由保留的事件重建，舊 DB 檔案請一併移出工作區。

主控本身故障時接收與回傳會停止；應由現場的服務管理機制重新啟動。重啟恢復不能保證當下那筆工作仍趕得上原始期限。

## 備份、MiNiFi 與統計

原始備份位置為 `backup_root/YYYY-MM-DD/job_id/`。結果保存在同一工作下的 `ai_result/returned`、`ai_result/processed` 與 `manifest.json`；匯出成功的 processed CSV 移到 `ai_result/exported`。`DONE` 表示 SPI 回傳與本機結果備份完成，MiNiFi 匯出可能仍在等待。

MiNiFi 只收集輸出目錄中的 `*.csv`，忽略 `.tmp`。背景程序依明確選定的結果 manifest 匯出，不掃描未選定的推論嘗試。NG 中無缺陷名稱的全 23 保護列標記 `AI_SKIP`。輸出完成不代表 MiNiFi 已上傳；上傳與下游資料庫由 MiNiFi 負責。部分輸出失敗後重試可能重複已被收走的 CSV，下游須依穩定 UUID 去重／upsert。

```bat
spi_statistics.bat --output reports
spi_statistics.bat --date 2026-11-09 --output reports
```

統計合併工作區與 `paths.history_root` 的不可變事件，同一 job/revision 只計算一次；結果 manifest 使用仍存在的工作檔或歷史副本。不依賴 DB 補登進度，也不修改工作狀態。輸出 jobs.csv 與 summary.csv；日期以 AI 首次觀測的台灣日期篩選。

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

不需要 GPU 就能執行設定、CSV、判定、期限、DB lock、恢復與子程序監督測試。真實程序整合測試使用假模型；不代表實際權重精度或 Windows／SMB 六秒達成率已驗證。要在 macOS 執行真實 CPU 模型可安裝 `--extra inference --extra mac`。

本分支不保留舊 API／scanner／三 Worker 相容入口，也不遷移舊 pipeline.sqlite3。切換前停止舊程式、保留舊資料，並確認輸入目錄只包含要由新版處理的工作；不要同時啟動兩種架構。
