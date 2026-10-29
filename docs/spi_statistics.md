# SPI 離線統計工具

將一台機器的現場資料放入同一個資料夾，以 Python 匯出適合 Excel、Power BI 或 Python 分析的 CSV。遞迴讀取子資料夾，不必把原本的備份結構攤平。

本工具只使用 Python 標準函式庫，支援 Python 3.10 以上，可獨立複製 `app/tools/spi_statistics.py` 執行。專案內的 Windows 啟動器沿用專案的 Python 3.12 選擇方式。

工具不啟動模型、呼叫機台、修改設定或搬動輸入檔；SQLite 先連同 WAL 複製至暫存目錄再查詢，避免 SQLite 在原資料夾產生共享記憶體檔。DB 裡記載的遠端或絕對路徑只拿來比對，不會前往讀取。

## 1. 準備資料

每次執行只處理**同一台機器**的資料；多台機器請分開執行，指定不同 `--machine-id`，之後依此欄位合併報表。同一個資料夾內相同 `job_id` 視為同一工作，不可混入獨立重播實驗或不同機台的相同 ID。

範例：

```text
statistics_input/
  ai_server.json                          # 當時產線設定，選用
  pipeline.sqlite3                        # 完整時間資料，建議提供
  logs/
    system.ingest
    system.inference
    system.publisher
    system.publisher.1                    # 輪替檔也會讀取
    log.csv                               # 若產線使用舊版 /process
    log.csv.1
  backup/
    2026-10-21/
      20261021111223/
        board.CSV                         # 原始 CSV，可保留原檔名
        image1.jpg                        # 要統計影像數時須提供完整影像
        image2.jpg
        ai_result/
          manifest.json
  events.csv                              # 機台／OP 時間與數量，選用
```

最少只要放入 manifest，即可取得工作 ID、產生結果時間、正常／保護性結果、原因、22／23 列數與模型請求耗時。JSON 檔案可以改名，靠內容識別；相同內容的複本不重複計算。

需要補充的資訊與來源：

| 要分析的資料 | 需要提供 |
|---|---|
| 完整分階段時間 | `pipeline.sqlite3`，或一筆工作狀態的 `job_state.json` |
| 資料完整時間、回傳時間、回傳原因 | `system.ingest`、`system.publisher` 及輪替檔可補充；只有 manifest 無法還原這些時間 |
| 歷史影像閾值 | 當時 `ai_server.json`，以 `--config` 指定；或 `--image-threshold` 明確指定假設值 |
| 影像數 | **完整**的原始 CSV／影像備份，保留工作 ID 資料夾；或 `events.csv` 的 `image_count`；超閾值 manifest 的錯誤文字也可能記載數量 |
| SPI Alarm 數 | `events.csv` 的 `spi_alarm_count`；不會把 CSV 列數自動當成 Alarm 數 |
| 機台送出、讀取、OP 確認時間 | `events.csv` |

從運行中的 SQLite 蒐集資料時，請用 SQLite 備份功能產生一致的單檔快照；或停止相關 Worker 後再複製。若提供的是既有 WAL 模式檔案組，請保留相同時間點的 `.sqlite3` 與 `.sqlite3-wal`，勿只複製主檔而漏掉尚未 checkpoint 的紀錄。工具會讀取提供的 WAL，但無法修復蒐集時已不一致的檔案組。

原始影像備份若漏複製，會造成影像數低估。工具會忽略 `ai_result` 下的產物，且當影像少於 manifest 列數時標示可能不完整、不採用該影像數；這不是對完整性的全面保證，仍須提供完整備份。若只想節省影像複製量，可先取得每份工作的正確影像數，填入 `events.csv`。

## 2. 執行

在專案根目錄執行，路徑有空白時保留雙引號：

```bat
05_spi_statistics.bat --input "D:\SPI_Statistics\SPI01" --output "D:\SPI_Statistics\reports\SPI01" --machine-id SPI01 --image-threshold 60
```

若使用當時產線設定：

```bat
05_spi_statistics.bat --input "D:\SPI_Statistics\SPI01" --output "D:\SPI_Statistics\reports\SPI01" --machine-id SPI01 --config "D:\SPI_Statistics\SPI01\ai_server.json"
```

也可以在專案根目錄自行建立 `statistics_input`，放入資料後直接雙擊 `05_spi_statistics.bat`；結果輸出至 `statistics_output`。雙擊模式**沒有預設閾值**，若要補足閾值請改用上面的參數。

跨平台或獨立執行：

```bash
python app/tools/spi_statistics.py --input ./statistics_input --output ./statistics_output --machine-id SPI01 --image-threshold 60
```

`--image-threshold` 與 `--config` 同時指定時，前者優先作為缺漏值的補值。manifest 的超閾值錯誤文字或事件 CSV 提供的逐筆閾值，優先於全域補值；若歷史證據互相衝突，會寫入 `issues.csv`。不會默默採用目前開發機的設定或預設 60。

輸出路徑可以是輸入資料夾的子目錄，掃描會排除該輸出目錄。輸出不能等於輸入，亦不能是輸入的父目錄。再次執行會覆寫六份同名報表；輸入資料不變。

## 3. 輸出檔案

全部使用 UTF-8 BOM，讓 Excel 正常開啟中文；時間為帶時區的 UTC+8，耗時欄位以秒為單位，模型 `request_ms` 以毫秒為單位。

| 檔案 | 用途 |
|---|---|
| `jobs.csv` | 每個工作一列，含時間、耗時、數量、閾值、結果、來源、缺漏與問題 |
| `summary.csv` | 全部及每日統計：樣本數、平均、P50／P95／P99、最大值、超過 3／5 秒比例、超閾值比例 |
| `image_buckets.csv` | 按 0、1–10、11–30、31–60、61–100、101+、unknown 分組，含各組耗時統計；分組固定，不隨閾值變動 |
| `model_timings.csv` | 每個工作、每個模型一列，方便做模型耗時圖 |
| `manifest_candidates.csv` | 所有不同內容的 manifest、指紋及來源，供查核重試與結果版本 |
| `issues.csv` | 讀取錯誤、候選結果不明、欄位衝突、負耗時等問題 |

`jobs.csv` 的重要欄位：

| 欄位 | 定義 |
|---|---|
| `ready_at` | 資料完整／穩定性驗證通過；不是首次看到檔案，也不是本機備份完成 |
| `inference_started_at` | Worker 首次取得推論工作；若曾重試，後續處理區間包含重試等待 |
| `result_ready_at` | 結果提交 SQLite 的時間；不以 manifest 時間冒充 |
| `manifest_created_at` | manifest 建立過程中記錄的時間；不是機台回傳完成時間 |
| `primary_returned_at` | 最後一份 Primary CSV 發布完成；不是 SPI 讀取時間 |
| `waiting_s` | `inference_started_at - ready_at`，含本機備份與排隊 |
| `inference_processing_s` | `result_ready_at - inference_started_at`，含模型通訊、分類、結果產生與可能的重試 |
| `publish_s` | `primary_returned_at - result_ready_at`，含發布等待與寫檔 |
| `internal_total_s` | Pipeline：`primary_returned_at - ready_at`；舊版：system log 的 `primary_return_latency_ms / 1000` |
| `request_total_s` | 舊版 CSV 的 `request_latency_ms / 1000`，含備份；與內部回傳耗時分開 |
| `pre_ready_s` | `ready_at - spi_sent_at`，含前段傳輸、穩定等待與驗證 |
| `machine_wait_s` | `spi_read_at - spi_sent_at` |
| `read_after_write_s` | `spi_read_at - primary_returned_at` |
| `late_after_op_s` | `spi_read_at - op_confirm_at`，正值代表晚於 OP，負值代表早於 OP |
| `late_after_op` | 晚於 OP 為 1，否則 0，缺少任何一個時間則空白 |
| `outcome`／`reason` | 正常結果或 fallback 的原因；未證明已發布時只是產生的結果 |
| `state_reason` | 最新 SQLite 狀態原因，可能是回傳後的備份錯誤，與決策原因分開 |
| `publication_evidence` | `primary_timestamp`、`published_backup`、`legacy_primary_latency` 或 `not_recorded` |
| `row_count`／`pass_count`／`fail_count` | manifest 結果列數及 22／23 列數；不代表板數或 Alarm 數 |
| `image_count`／`spi_alarm_count` | 影像數與另行提供的機台 Alarm 數，兩者保持分開 |
| `image_threshold_exceeded` | 超閾值為 1、未超過為 0，無法判斷則空白；等於閾值不算超過 |
| `threshold_source`／`threshold_evidence` | 閾值的資料來源、用數量比較或由超閾值原因確認 |
| `manifest_selection` | 最終候選的選擇依據；`single_unverified` 表示唯一但未證明已發布，`ambiguous` 表示無法判定 |
| `manifest_fingerprint` | 對照候選清單的內容指紋 |
| `field_sources` | JSON 格式的逐欄來源；其餘 `*_sources` 保留檔案路徑或行號 |
| `missing_fields`／`data_issues` | 缺少的時間／數量欄位及這份工作的資料問題 |

一份工作可能包含多份 CSV／多塊板。此工具以 job 為單位去重，不能直接稱為板級時間；`primary_returned_at` 是全部 CSV 的最後完成時間。

## 4. 統計口徑

- 超閾值比例：`threshold_exceeded_jobs / threshold_known_jobs × 100`。無法判斷的工作另外列在 `threshold_unknown_jobs`，並提供 `threshold_coverage_pct`。**不是**將未知工作視為未超過。
- 影像加權比例：在「影像數與是否超閾值都已知」的工作內，超閾值工作的影像總數除以該樣本的影像總數。樣本工作數為 `image_weighted_known_jobs`；分母為零時空白。
- `threshold_skip_result_jobs` 是已識別結果中原因為 `images_exceed_threshold` 的數量；`threshold_skip_published_jobs` 額外要求有發布證據。前者不等於已回傳機台的數量。
- 每個耗時統計都有自己的 `*_n`，分位數與超過 3／5 秒比例只使用該欄有效樣本；比較採嚴格大於。P50／P95／P99 採排序後線性插值。
- 正常結果不代表整份工作 OK；`normal_result_jobs` 表示正常跑完決策，仍可能包含 AI 判 NG。fallback NG 與模型判 NG 不混為一談。
- `total_jobs` 是提供資料中可識別的唯一工作數，包含尚未回傳與缺少結果的工作。沒有出現在輸入證據裡的生產板，工具無法得知；此分母不是機台總產量。
- 日期依 `ready_at`、舊版 request 開始、機台送出、manifest 建立的順序選取，並記錄 `report_date_source`。無時間則為 `unknown`，不從資料夾名稱臆測日期。
- 多模型並行，`model_timings.csv` 是各模型的請求牆鐘耗時，不能相加當成整體推論時間，也不是純 GPU 運算時間。舊版 log 人工補出的零耗時不輸出成模型測量值。
- 最終 manifest 優先依 SQLite 指向的檔案識別，其次 `ai_result` 發布備份及可唯一匹配的 Publisher 證據。無法確認的多份候選不取最新一份猜測，而是保留候選清單、清空結果列數與模型耗時。
- 舊版 log 多次請求相同 job 時，以最新 request CSV 為明細，`legacy_request_count` 保留請求數。多次請求無法安全對應 Publisher 耗時時不拼接。相同 job 同時出現 Pipeline 與 `/process` 證據時，只採 Pipeline 並列出警示。

## 5. 選用的 events.csv

可放在輸入資料夾任意位置自動讀取，或以 `--events "D:\events.csv"` 指定。欄名固定為英文，`job_id` 必填，其餘可留空；CSV 必須為 UTF-8（可含 BOM）。

```csv
machine_id,job_id,spi_sent_at,spi_read_at,op_confirm_at,spi_alarm_count,image_count,image_threshold
SPI01,20261021111223,2026-10-21T14:07:30.000+08:00,2026-10-21T14:07:36.800+08:00,2026-10-21T14:07:35.500+08:00,8,8,60
```

上述為欄位示例，不是正式現場紀錄。`spi_sent_at` 建議固定採機台「本次資料送出完成」事件；若現場選擇開始送出時間，整批資料要保持一致，避免混用起點。跨機器比較前先核對時鐘。

所有 ISO 時間必須帶時區（如 `+08:00` 或 `Z`），也接受 Unix 秒。沒有時區、負耗時與非有限數字不會猜測補值；負耗時會列入問題清單並排除該區間。`late_after_op_s` 本身允許負值，代表 AI 結果較早被讀取。

同一 job 可以有多列以補齊不同欄位，但同一欄有衝突值時不採用外部值並記錄問題。機台標籤若填寫，必須與 `--machine-id` 一致。事件資料與系統證據衝突時保留系統證據並標示問題。

## 6. 建議圖表與檢查

1. 用 `jobs.csv` 畫 `waiting_s`、`inference_processing_s`、`publish_s` 堆疊圖，檢查延遲所在階段。
2. 用 `image_buckets.csv` 比較各影像數區間的 `internal_total_s_p95`；搭配 `*_n` 確認樣本數。
3. 用 `summary.csv` 的每日超閾值比例、coverage 與未知數量，避免資料缺漏造成比例失真。
4. 有機台事件後，畫 `late_after_op_s` 分布，並列出 `late_after_op=1` 的工作供現場核對。

執行完先看 `issues.csv` 與 `missing_fields`，再解讀統計。程式回傳碼：0＝有工作且沒有讀取／正規化錯誤（仍可能有 warning），1＝參數或輸入錯誤／沒有工作，2＝已輸出報表但有部分資料錯誤。錯誤詳情保留在 CSV；修正輸入後重跑即可。
