# SPI AI 單一主控

流程固定為：掃描未完成 CSV → 備份 → OBB 推論 → SPI 回傳 → 來源加上 `_DONE` → 整理 MiNiFi。

來源 CSV 檔名是完成判斷依據。沒有 SQLite、state_root、history_root、事件日誌或跨重啟恢復排程；每次處理的時間與結果保存為備份目錄內的 `run.json`。

## 啟動

需要 Python 3.12。Windows / NVIDIA AIPC：

```bat
setup.bat
start_spi_ai.bat
```

跨平台入口：

```sh
python -m app.main --check-config
python -m app.main
python -m app.main --once
python -m app.main --once --date 2026-11-09
```

正常模式持續掃描，工作依發現順序逐筆處理。`--once` 處理本次掃描找到的一批工作，最後再嘗試一次尚未完成的改名／MiNiFi 匯出；有輸入失敗、回傳失敗、部分回傳、不明回傳或改名未完成時回傳 exit code 2。啟動或掃描失敗回傳 1。`--date` 僅搭配 `--once`，不指定時掃描所有有效日期。

主要程式：

```text
app/application/controller.py       # 記憶體佇列、處理循環及改名／匯出重試
app/application/ingest/              # CSV、影像驗證與原始備份
app/application/inference/          # OBB 與 CSV 逐列判定
app/application/operations.py        # 固定回傳檔、SPI 發布與來源改名
app/infrastructure/run_record.py    # 每次處理的 JSON 報告
app/minifi.py                       # 結果轉為 MiNiFi CSV
```

## 掃描與手動重跑

- 掃描 `paths.watch_root` 直接下一層的 `YYYYMMDDHHMMSS` 資料夾；也接受直接指定一個時間戳資料夾。
- **所有沒有 `_DONE` 後綴的 CSV 都是待處理資料，包含啟動前既有的檔案及舊日期資料。** 沒有啟動略過清單，也不會查詢備份或 JSON 來排除工作。
- `abc.csv` 成功回傳後，來源改為 **`abc_DONE.csv`**。`_DONE` 大小寫不敏感；CSV 副檔名保留。`abc_DONE1.csv` 不是完成標記。
- 同資料夾多份 CSV 逐份回傳、逐份標記；只讀取本次認領的 CSV，後來新增的 CSV 留給下一批。
- 手動將 `abc_DONE.csv` 改回 `abc.csv` 即可重跑，可以在程式運行中或停機時操作。重跑會重用相同內容的平鋪原始備份，並更新 `ai_result` 中該份 CSV 的最新結果。原始檔內容不同時報錯，不覆寫既有原始備份。
- 掃描操作失敗會記錄警告並稍後重掃。`discovery_timeout_seconds` 是單次目錄讀取操作上限，並非等待機台產生資料的期限。正常一輪掃描結束後，依 `scan_interval_seconds` 再掃一次。

來源共享目錄需要讀取及改名權限。輸入內 CSV／影像的名稱對應規則維持原契約，來源改成 `_DONE` 不會更改回傳 CSV 的原檔名。

## 設定

使用 `config/ai_server.json`，可透過 `--config` 或 `AI_CONFIG_PATH` 指定另一份完整設定。相對路徑以專案根目錄解析，設定修改後重啟生效。

| 區塊 | 用途 |
|---|---|
| identity | site、factory、line、station_id，供結果與 MiNiFi 使用 |
| paths | watch_root、external_output_root、backup_output_root |
| processing | 影像數上限、副檔名、影像命名、檔案穩定等待與就緒上限 |
| inference.obb | 權重、裝置、批次、暖機／推論上限與各類別門檻 |
| delivery | 每次處理中，每份 CSV 的回傳次數、間隔與單次操作上限 |
| runtime | 掃描、IO、背景操作上限與重試間隔 |
| output | SPI 回傳是否保留 job 資料夾 |
| minifi | enabled、output_path |
| logging | 系統日誌目錄及檔名 |

`paths` 三個資料目錄與 MiNiFi 輸出目錄必須互相分開，不可互相包含。`database`、`paths.state_root`、`paths.history_root`、`minifi.poll_seconds` 已移除；未知設定會被拒絕。

## 每次處理

1. 發現未完成 CSV 後放入記憶體佇列，同一資料夾同時只處理一批。模型暖機與排隊時間獨立記錄。
2. 等待選定的 CSV 與影像穩定，驗證並保存不可變 CSV 快照。預設影像名稱為 `{csv_stem}_{component_name}_{Array_id}_{Pad_id}.jpg`，保留 CSV 字串中的前導零。
3. 在獨立備份位置串流複製、比對 SHA-256 並驗證影像可解碼，再交給 AI。來源若仍在變動則重新等待。
4. OBB 外觀判斷後，再以原 CSV 每列的 volume、area、height、shift 門檻判定，見 [判定規則](docs/sx_obb_inference.md)。
5. 固定並備份本次結果與 SPI 回傳 CSV，逐份回傳。回傳只更新原 CSV 的 is_pass，保留檔名、欄位、編碼、列序及其他值。重試只複製同一份固定結果，不重新推論。
6. 確認某份 CSV 回傳成功後，核對來源內容仍與本次快照相同，再改成 `_DONE.csv`。已有同名完成檔時不覆寫。來源已變動、改名權限不足等狀況寫入 JSON 與日誌；本次運行只重試改名，不重送已確認回傳的結果。
7. 將已確認回傳的逐份結果轉為 MiNiFi CSV。同一時間戳資料夾若仍有待改名或待匯出的結果，先完成收尾再接收下一批；其他資料夾繼續處理。部分回傳成功時，成功的 CSV 仍可獨立匯出。匯出失敗在本次運行內重試，讀取固定備份，不重新推論或回傳。

`processing.folder_images_num_threshold` 預設 **60**，以整個來源工作資料夾符合副檔名的實際影像數判斷。60 張照常推論，61 張直接全 23。完整 CSV 等不到影像、模型失敗或推論逾時也使用全 23 保護結果；這些結果確認回傳後同樣標記 `_DONE`，原因保存在 JSON。無法完整解析 CSV 時不捏造結果、不標記完成。保護結果的影像備份為盡力保存，若不完整會記錄 `image_backup_status=incomplete`。

檔案穩定只是完成推估，機台需保留來源直到處理完成。

## 回傳、重試與重啟

`delivery.max_attempts` 包含第一次回傳，預設為 3。每次嘗試的開始、結束、耗時、錯誤與結果都保存在 `run.json`。

| 狀況 | 處理 |
|---|---|
| 確認回傳目錄的原子改名成功 | 記為 RETURNED，嘗試來源加 `_DONE` |
| 明確未送達，例如權限拒絕 | 本次依設定重試；仍失敗則不加 `_DONE` |
| 可能已送達，但沒有收到操作完成確認 | 記為 UNKNOWN，本次運行不自動重送，不加 `_DONE` |
| 回傳成功但來源改名失敗 | 記為 MARK_PENDING，本次只重試來源改名 |

已失敗且仍未標記的 CSV，會在 `runtime.background_retry_seconds` 後再次成為待處理工作。UNKNOWN 與已回傳但未標記的同一份來源，在本次運行內暫停重新認領；來源移除／更換後可重新接收。

**重啟後只看來源檔名。** 未加 `_DONE` 的 CSV 都會重新處理，因此「已回傳、尚未改名就當機」或 UNKNOWN 的工作有重複回傳的可能。JSON 只供查閱與統計，不用來恢復或阻止輸入工作。已加 `_DONE` 的工作不會因 MiNiFi 尚未匯出而重新推論；重啟前未完成的背景匯出需依備份與 JSON 核對處理。

回傳成功只代表輸出目錄寫入成功，目前沒有 SPI 業務層的接收／接受 ACK。MiNiFi 輸出成功也不代表已上傳到下游。

單一操作仍受監督，以免 SMB／模型永久卡住：

| 設定 | 預設秒數 |
|---|---:|
| processing.input_ready_timeout_seconds | 30 |
| runtime.discovery_timeout_seconds | 5 |
| runtime.io_startup_timeout_seconds | 30 |
| runtime.io_timeout_seconds | 30 |
| inference.obb.startup_timeout_seconds | 120 |
| inference.obb.inference_timeout_seconds | 120 |
| delivery.attempt_timeout_seconds | 10 |
| runtime.background_timeout_seconds | 30 |

背景重試間隔由 `runtime.background_retry_seconds` 控制。每個 input 路徑只允許一個主控；OS 鎖位於系統暫存區。執行中的 CSV 與結果暫存也使用系統暫存區，提交到備份後清除，不載入舊的工作狀態。

## 備份與時間紀錄

```text
backup_output_root/
  YYYY-MM-DD/job_id/
    board.csv                  # 原始 CSV，維持原檔名與內容
    board_J001_001_0.jpg        # 原始影像直接平鋪
    board_J001_001_1.jpg
    ai_result/
      run.json                 # 最近一批的階段時間、設定、回傳與改名紀錄
      manifest.json            # 最近一批的模型資訊、逐列結果與 run_id
      returned/                # 各 CSV 最新的固定 SPI 回傳內容
      processed/               # 待匯出的 processed CSV
      exported/                # 已整理成 MiNiFi 格式的 processed CSV 備份
```

每次重跑仍有新的 run_id，僅保存在 JSON 欄位，不新增 run_id、input_csv 或 raw 資料夾。CSV 快照與本次推論使用的資料放在系統暫存區，完成後清除；備份時間戳資料夾只平鋪原始 CSV、影像與 ai_result。`ai_result/run.json` 與 manifest 保存最近一批處理紀錄，該批 CSV 的 returned／processed／exported 更新為最新結果，其他 CSV 的結果保留。`run.json` 記錄首次發現、開始處理、CSV 就緒、原始備份起訖、推論起訖、結果就緒、逐份回傳嘗試／完成、來源改名與 MiNiFi 匯出起訖。時間同時提供 epoch 秒與台灣時間 ISO 字串；JSON 透過暫存檔與原子替換更新。

MiNiFi 只收集輸出目錄中的 `*.csv`，忽略 `.tmp`。同一次執行重試使用穩定 UUID，手動重跑使用新的 run_id／UUID 與輸出名稱。NG 中沒有缺陷名稱的全 23 保護列標記 `AI_SKIP`。部分輸出失敗後重試可能再次輸出已被收走的 CSV，下游應依 UUID 去重／upsert。

統計從上述備份 JSON 與 manifest 產生：

```bat
spi_statistics.bat --output reports
spi_statistics.bat --date 2026-11-09 --output reports
```

輸出 jobs.csv、deliveries.csv、summary.csv；每個時間戳資料夾只統計目前 run.json 中最近一批的紀錄，不累積先前重跑歷史；日期以該批 AI 首次觀測的台灣日期篩選。工具不會改動來源、執行推論或回傳。

## 舊版本切換

停止舊程式，換成新版設定並移除 database、state_root、history_root 與 minifi.poll_seconds。舊 SQLite、state/history 及既有備份可留存查閱，新版不會刪除它們。先前使用 run_id/input_csv/raw 的備份不會自動搬移；後續處理使用新的平鋪結構。

**舊版已完成但來源仍沒有 `_DONE` 的 CSV，新版會再次處理。** 切換前需核對並將不需重跑的來源改成 `_DONE.csv`。來源不需重跑時，不能只依舊版歷史紀錄判斷新版會自動略過。

## 模型格式與離線匯出

OBB 支援 `.pt`、`.onnx`、`.engine`，必須是七類 SPI OBB。`.engine` 需要 NVIDIA CUDA；固定批次尾批補圖後丟棄補圖結果。ONNX 固定批次由模型 shape 讀取。

```bat
convert_to_tensorrt.bat --model models/appearance/best.pt --output models/appearance/best.engine --device cuda:0 --batch 8 --half
```

輸出副檔名也可為 `.onnx`。若以 `--dynamic` 匯出 engine，設定 `engine_fixed_batch=false`。TensorRT engine 應在實際 AIPC 的 GPU／CUDA 環境產生與驗證。

## 開發驗證

```sh
uv sync --group dev
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

單元測試不需要 GPU；程序整合測試使用假模型，不能代替實際權重精度或 Windows／SMB 現場驗證。
