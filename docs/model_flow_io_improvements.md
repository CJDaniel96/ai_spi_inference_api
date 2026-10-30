# 模型串接與 I/O 改善

保留 SQLite、現有三段 Worker 與原始六秒期限。本次不更換產線權重，新增的面積比服務預設停用。

## 接收資料

一般案件先保存完整 CSV，再依本機 CSV 快照檢查共享資料夾的影像檔名、檔案類型與大小。
影像內容在複製時讀取一次，同時產生 SHA-256；之後在本機驗證副本內容與影像可解碼性。
若設定獨立 staging，第二份副本也從本機建立，不再重讀共享磁碟。

保留來源前後檔案集合、大小與修改時間檢查、CSV 快照比對、既有目的地內容衝突檢查，以及暫存資料夾完成後重新命名的發布方式。
這仍以機台輸出期間會更新檔案資訊為前提；沒有機台完成標記時，不能把檔案穩定觀察當成生產端完成的絕對證明。

新工作只有在本機原始副本驗證完成、轉為 READY 時才記錄 `ready_at`。
整體期限仍使用最初的 `first_seen_at + end_to_end_timeout_seconds`，不因複製、重試或重啟而重新計時。
超過 60 張的 CSV-only 路徑維持不解碼影像、不等待影像備份。

## 回傳後備份

正式 `app.pipeline publisher` 使用獨立、可監督的備份子程序。
主 Publisher 每輪只輪詢備份完成狀態，仍能接收下一筆 Primary 回傳，不等待前一筆備份完成。

- 子程序只處理本機 returned CSV、processed CSV 與 manifest，沒有 SQLite 連線。
- 父程序收到完成確認後才記錄 DONE。
- 單次嘗試上限為 `min(30 秒, worker_lease_seconds × 0.8)`，在認領租約到期前停止。
- 備份失敗沿用最多五次、5／10／20／40 秒延後重試；不重送已回傳的機台 CSV。
- 程序關閉或當機時，PRIMARY_RETURNED 與租約留在 SQLite，重啟後依租約恢復。
- `publisher --once` 會等待自己啟動的這一次有限時備份；長駐模式持續處理回傳。

直接建立 PublisherWorker 且未傳入 `finalize_io` 的同步呼叫方式仍保留，供既有工具與單元測試使用。

## Detection → Segmentation

新增 `python -m app.area_service`，預設位址 `127.0.0.1:8003`。
服務啟動時載入並暖機兩個模型。每筆工作只呼叫一次 HTTP，影像只解碼一次：

1. Detection 找出 `Solder_Paste`，選擇最接近影像中心的物件。
2. 對有中心物件的影像做全圖 Segmentation，保留 Pad 的完整範圍，不自行裁切 ROI。
3. 以 Detection 框與分割物件框的 IoU 配對 `Solder_Paste`、`Pad`，要求正重疊；同分歧義回報無效。
4. 以原圖尺寸的遮罩計算 `cover% = Solder_Paste 像素數 / Pad 像素數 × 100`。

使用類別名稱解析模型內的數字 ID，不假設兩個模型的類別順序相同。
原圖尺寸遮罩由 `retina_masks=True` 取得；尺寸不符、缺少中心、遮罩缺漏、空面積等會產生 `None` 與原因。
在 `area_ratio.required=true` 時，任何必要影像結果無效，都會沿用整筆全 23 的政策。

服務回應含 `results`、`details`、Detection／Segmentation 各步驟耗時、權重指紋與裝置資訊，並輸出到服務日誌。
既有 processed CSV 合併面積比；服務詳細資料不會加到機台 CSV。

新面積比直接使用兩個遮罩的同尺度像素面積，不套用舊公式的 `0.8246` 換算係數。
判定仍使用 `defect_rules.high_cover_threshold=180.0`，只有大於 180% 才命中 high cover；既有其他啟用規則照常執行。
Pad 的標註需代表作為分母的完整 Pad 區域。如果模型只分割裸露的 Pad，分母定義會不同，需在產線啟用前核對。
最接近影像中心與 IoU 配對策略也需以實際影像確認，不能僅憑假模型測試判定精度。

### 啟用方式

1. 將實際權重放在專案 `models/` 下，修改 `config/yolo_area.json` 的兩個路徑。
   預留路徑為 `models/area/center.pt`、`models/area/segment.pt`，也可使用 `.onnx` 或 `.engine`。
2. 核對模型內有 `Solder_Paste`／`Pad` 類別，設定實際 device、imgsz、batch。
   固定批次 TensorRT 引擎需設定 `fixed_batch_size=true` 並填入正確 batch_size，尾批次會補黑圖並丟棄補圖結果。
3. 先執行 `start_yolo_area_service.bat`，檢查 `/health` 的 `model_ready=true`，再以保存的產線資料確認配對、遮罩與面積比。
4. 確認後才將 `config/ai_server.json` 中 `area_ratio.enabled` 改為 `true`；之後 `start_model_services.bat` 也會啟動此服務。
   既有 anomaly／distance 是否停用，需依實際需要的缺陷規則設定；啟用新服務不會自動停用它們。
5. 重新啟動 Worker，量測包含接收、排隊、兩模型與共享磁碟回寫的完整耗時。

權重未提供前維持 `enabled=false`。缺少權重或類別不符時服務會保持 model_ready=false，拒絕推論。

### 期限與並行

Stage 02 將扣除發布預留時間後的截止時間，透過 `X-SPI-Deadline-At` 傳給新服務。
服務會在每次模型呼叫前後檢查期限，過期時不啟動下一個模型、不回傳晚到結果。
舊服務可忽略此 Header，原有 client timeout 與結果提交期限檢查仍保留。

新服務只允許一筆 GPU 工作執行，忙碌時回傳 503，避免額外的 GPU 等待佇列。
維持一個 HTTP worker；增加 Uvicorn worker 會增加模型副本與 GPU 記憶體需求。
HTTP timeout 無法中止已經開始的 GPU 核心；即使模型卡住，外部 Publisher 仍能依期限回傳全 23。

## 驗證範圍

自動測試使用暫存資料夾與假模型，涵蓋來源讀取次數、來源變更、損壞影像、本機副本、備份未完成時下一筆回傳、備份失敗恢復、雙模型順序、共用影像、缺漏遮罩、過期期限、固定批次，以及 180% 判定邊界。
尚需在 Windows／SMB 與實際 GPU 權重上量測六秒達成率、最壞耗時及判定精度。

本次驗證：`pytest -q` 共 437 項通過；`ruff check app tests`、`ruff format --check app tests` 與 `git diff --check` 通過。
全倉 Ruff 另有既有問題：`tmp_artifacts/build_deployment_manual.py` 的 148 項檢查錯誤，以及該檔與 `patchcore_inference.py` 的格式差異。本次未修改這兩個檔案。
