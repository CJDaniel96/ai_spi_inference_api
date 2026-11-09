# SX OBB 推論與判定

目前正式設定為 `inference.mode="yolo_obb"`。02 管理一個以 spawn 啟動的常駐模型子程序，啟動時載入、暖機並驗證類別，之後重複使用同一模型。每次只接受一筆工作，透過本機 IPC 傳送影像路徑與結構化結果；無需啟動模型 HTTP server。

## 啟動與權重

1. 將實際權重放到本機，例如 `models/appearance/best.pt`。
2. 修改 `config/ai_server.json` 的 `inference.obb.model_path`。
3. 先啟動 03，再啟動 02；看到 `event=obb.ready` 後啟動 01。

```json
"inference": {
  "mode": "yolo_obb",
  "site_id": "SX",
  "obb": {
    "model_path": "models/appearance/best.pt",
    "device": "cuda:0",
    "imgsz": 640,
    "batch_size": 8,
    "engine_fixed_batch": true,
    "conf": 0.25,
    "class_thresholds": {
      "Solder_Paste": 0.7,
      "Stain": 0.25,
      "Missing": 0.25,
      "Insufficient": 0.25,
      "Slump": 0.25,
      "Foreign_object": 0.25,
      "Bridge": 0.25
    }
  }
}
```

- `.pt` 使用 Ultralytics／PyTorch；`.onnx` 使用 Ultralytics／ONNX Runtime；`.engine` 使用 Ultralytics／TensorRT。
- 固定批次 TensorRT 先以 batch=8 執行，尾批不足時補黑圖，補圖結果不納入判定。動態批次引擎可設定 `engine_fixed_batch=false`。
- ONNX 固定批次從模型 NCHW 輸入維度讀取並自動補齊；動態批次使用設定的 `batch_size`。
- `imgsz`、TensorRT batch 與裝置需符合實際匯出的模型；副檔名選擇後端不代表自動轉換權重。
- ONNX 需對應的 ONNX／ONNX Runtime 套件；TensorRT 需 CUDA 裝置及相容 runtime。類別依名稱解析，不假設固定的 class ID 順序。
- 模型類別名稱 `Foreign_Object` 相容於設定中的 `Foreign_object`；啟動驗證與推論結果會統一為 `Foreign_object`，套用同一個 NG 信心門檻。設定檔與輸出紀錄維持使用 `Foreign_object`。缺少、重複或其他未知類別仍會啟動失敗。
- 範例權重路徑是預留位置，專案未包含權重；缺檔會明確啟動失敗，不自動下載其他模型。

## 固定判定順序

`YOLO OBB → 體積 → 面積 → 高度 → X/Y 位移`。

每列在第一個失敗的規則停止，作為主要缺陷；全部通過才回傳 `is_pass=22`，不通過回傳 `23`。

外觀檢查先使用模型推論 `conf=0.25` 取得候選框，再依 `class_thresholds` 各自篩選，門檻比較為 `>=`。
篩選後需要**剛好一個 Solder_Paste 框，且沒有任何 NG 框**。例如 OK 框信心為 0.71／0.50／0.30 時，後兩個被排除，可以繼續體積檢查；若有兩個 OK 框皆達 0.7，則外觀不通過。
模型成功回傳零框屬外觀 NG，與缺少影像結果或模型失敗分開記錄。
NMS 保持 class-aware，避免 OK 與 NG 因不同類別的重疊而互相抑制。若結果碰到 `max_det` 上限，視為可能遭截斷的無效推論，使用保護性回傳。

## 逐列使用 CSV 門檻

量測門檻直接取機台 CSV 的同一列。已取消固定元件閾值 JSON 與 `rules_path` 設定；執行時不讀取 Excel，也不因元件名稱套用其他門檻。同一元件在不同列出現不同限制時，各列分別判定。

| 檢查 | 實測值 | 來源門檻 | 單位 |
| --- | --- | --- | --- |
| 體積 | `insp_vol` | `vol_l_ng`、`vol_h_ng` | % |
| 面積 | `insp_area` | `area_l_ng`、`area_h_ng` | % |
| 高度 | `insp_hei` | `hei_l_ng`、`hei_h_ng` | % |
| X 位移 | `insp_x` | `shift_x_ng` | mm |
| Y 位移 | `insp_y` | `shift_y_ng` | mm |

三種百分比量測均先正規化區間，再包含上下界比較：

```text
lower = min(l_ng, h_ng)
upper = max(l_ng, h_ng)
PASS = lower <= insp <= upper
```

例如 `vol_l_ng=150`、`vol_h_ng=50` 時，實際判定區間為 `[50, 150]`；與兩欄正常排序時結果相同。上下限相等時，只接受實測值恰好等於該值。

位移需要 X、Y **同時**通過：`abs(insp_x) <= shift_x_ng` 且 `abs(insp_y) <= shift_y_ng`，兩軸分別使用自己的限制。位移門檻需為非負數；零門檻只接受零位移。高度直接使用百分比，不做 Stencil 換算。

正規化只影響內部比較，不交換或改寫原始 CSV 欄位。缺失、空白、非數值、NaN 或無限大的必要量測／門檻會觸發保護性全 23，不回退到文件值或預設閾值。

## 期限、異常與追蹤

原有六秒期限及回傳預留時間繼續生效。02 把扣除發布預留時間後的絕對期限傳給模型程序，父程序限制等待，子程序在每批推論前後檢查期限。逾時或取消等待會回收子程序，後續工作前重建模型。
Worker 每次認領工作前先確認模型已暖機；冷啟動恢復期間不持有工作租約，03 可繼續接管到期工作。若暖機後程序仍在推論提交前異常退出，該筆的重建嘗試仍受原截止時間限制。

模型異常、結果缺漏、無效框資料、必要量測或門檻缺失／非數值等，會產生整筆全 23 的保護性結果；03 繼續負責截止接管。成功取得零框則是正常外觀 NG。既有超過 60 張影像的全 23 路徑維持原政策。

機台 CSV 仍只更新 `is_pass`。processed CSV 增加：

- `ai_failed_rule`、`ai_reason`、`ai_limit_source=csv`。
- `obb_raw_detections`、`obb_retained_detections`、OK／NG 框數，含框座標與各框信心。
- `ai_rule_checks`：實際執行的規則、量測值、來源門檻欄位與值、正規化後的上下限及 `limits_reversed` 標記。
- `ai_site_id`、`ai_rule_version`、`ai_model_version`。

manifest 的可選 `inspection_metadata` 保存 CSV 欄位映射、區間正規化策略、類別門檻、規則版本與權重 SHA-256；03 備份 manifest 時一併保存。`rule_version` 識別演算法及 OBB 閾值，逐列數值門檻則保留於原始 CSV 與 `ai_rule_checks`。舊 manifest 仍可讀取。

## 設定修改與相容模式

OBB 類別門檻在 `config/ai_server.json` 設定，修改後重啟 02 生效；量測限制直接讀每筆來源 CSV，不需維護另一份元件設定。尚未加入管理 Web。
規則判定集中於 `component_judge.py`；模型後端及程序管理位於 `app/infrastructure/inference/`。

舊 HTTP 推論保留為明確的 `legacy_http` 模式，設定範例為 `config/ai_server.legacy.example.json`。`/process` 拒絕 native OBB 設定，避免誤走舊判定流程；`start_model_services.bat` 在 native 模式只提示啟動 02。

自動驗證使用假模型與實際 spawn／IPC，涵蓋權重格式分流、固定／動態批次、類別過濾、逐列限制、正反向／相等區間、邊界、缺失門檻、逾時與重啟、跨階段回傳及備份。實際 GPU 權重的精度與六秒達成率仍需現場驗證。
