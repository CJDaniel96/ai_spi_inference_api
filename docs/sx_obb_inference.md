# SX OBB 判定

單一主控使用常駐模型子程序執行七類 OBB：Solder_Paste、Stain、Missing、Insufficient、Slump、Foreign_object、Bridge。

推論前依 `inference.obb.center_crop_ratio` 取原圖中心區域：`0.6` 保留原寬與原高各 60%，`1.0` 不裁切，省略欄位預設 `1.0`。尺寸取整數捨去且至少為 1 像素，起點取 `(原尺寸 - 裁切尺寸) // 2`。裁切後才送入 YOLO，三種模型格式共用相同前處理；備份仍保留完整原圖。

OBB 多邊形的每個點會加回該影像的裁切偏移，因此 processed CSV 內的 `obb_raw_detections` 與 `obb_retained_detections` 使用原始完整影像的像素座標。manifest 的 `inspection_metadata.preprocessing` 記錄比例、取整方式與座標空間，前處理設定也納入規則版本。

模型先使用 `inference.obb.conf` 過濾候選框，再使用 `class_thresholds` 的各類別門檻過濾。門檻必須不小於模型 conf，避免已在模型端丟棄的框無法恢復。

外觀以每張原始完整影像的幾何中心 `(寬 / 2, 高 / 2)` 選取 Solder_Paste，使用還原至原圖座標的旋轉四邊形判定。優先選取包含中心點（含邊界）的框，不要求框本身的中心完全對齊原圖中心。

若沒有框包含中心點，再使用 `inference.obb.center_tolerance_ratio` 尋找稍微偏移的框。預設 `0.05`，容許中心點到旋轉框最近邊緣的歐氏距離小於等於「原圖短邊 × 0.05」；例如 1000×800 影像容許 40 像素。比例須為有限數值 `0 <= ratio <= 0.5`，`0` 關閉彈性範圍。比例依原圖尺寸計算，與 `center_crop_ratio` 的裁切範圍分開。

1. 任何位置出現達門檻的其他 NG 類別，外觀即不通過。
2. 若有框包含中心點，必須恰好一個，其他框不再參與中心候選的選取。若沒有，則容許距離內必須恰好一個合格 Solder_Paste。即使全圖只有一個合格框，超出容許距離仍不通過；同一選取階段有多個候選也不通過，不以信心分數猜測目標。
3. 其他合格 Solder_Paste 框若與中心框有正面積交集，外觀不通過，包含小框被中心框完整包住的情況。僅邊或角接觸、交集面積為零時視為不重疊；不重疊的鄰近框忽略。鄰近框彼此重疊不影響中心目標。
4. 通過上述條件才繼續 CSV 數值檢查。無效或退化的合格 Solder_Paste 四邊形不能當作正常框，也不能直接忽略。

| 外觀 NG 原因 | 意義 |
|---|---|
| `missing_qualified_solder_paste` | 沒有達信心門檻的 Solder_Paste |
| `missing_center_solder_paste` | 有合格 Solder_Paste，但沒有框包含原圖中心或位於容許距離內 |
| `multiple_qualified_solder_paste` | 多個合格框都包含中心，或沒有包含中心的框、但容許距離內有多個候選；無法唯一選取 |
| `overlapping_qualified_solder_paste` | 唯一中心框與其他合格 Solder_Paste 框有正面積重疊 |
| `invalid_solder_paste_geometry` | 合格 Solder_Paste 框的幾何資料無效 |

其他合格 NG 類別沿用類別名稱作為原因；同時存在時優先於上述原因。

## 元件體積與面積判定

外觀通過後，從 SPI CSV 讀取 `component_name`、`insp_vol`、`insp_area`。元件名稱去除前後空白，依完整名稱比對 `components`，未列出的元件套用 `others`。空白元件名稱視為輸入無效。體積及面積都須落在各自的 `min_pct` 到 `max_pct` 之間（含邊界）。兩項都超限時，第一個失敗項目記為 `volume`，稽核保留兩項結果。

規則檔由 `component_rules_path` 指定，預設 [SX 元件規則](../config/rules/sx_component_judge.json)，在啟動時載入，修改後重啟生效。JSON 和 SPI CSV 都使用百分比刻度，`30%` 對應數值 `30`。啟動時會驗證 `others`、重複鍵與各上下限。

量測值必須是有限數值。必要資料缺漏、模型失敗、模型結果或原圖中心資料不完整／無效、超時時，整份工作走全 23 保護。`manifest.outcome` 記錄工作屬於正常判定或保護結果。

processed CSV 的 `ai_component_rule` 記錄實際使用的元件或 `others`，`ai_rule_checks` 記錄量測與實際上下限。`ai_rule_version` 的雜湊納入整份元件規則。SPI 回傳 CSV 只更新原始 `is_pass`（22 通過，23 不通過），保留欄位、編碼、列序與其他原始值。

### 從 Excel 更新規則

在維護環境安裝 `openpyxl`，從專案根目錄執行下列命令。匯入器會驗證整張表後才替換 JSON。預設使用活頁簿的作用中工作表，也可用 `--sheet Sheet1` 指定。

```bash
python -m pip install "openpyxl>=3.1,<4"
python -m app.tools.import_component_rules "SX SPI Component Judge Rules.xlsx" config/rules/sx_component_judge.json
python -m app.main --check-config
```

原始範本使用 `Component`、`OBB外觀`、`體積`、`面積` 四欄，接受 `30% ~ 170%` 這類文字範圍。拆分上下限時使用 `Component`、`OBB外觀`、`體積下限`、`體積上限`、`面積下限`、`面積上限`，四個門檻以 Excel 百分比儲存。兩種格式皆須包含一列 `others`，且 `OBB外觀` 填 `啟用`。

`ai_rule_checks` 的 `yolo_obb` 項目記錄 `image_center`、`center_count`、`center_selection_mode`、`center_tolerance_px`、`overlap_count`、`ignored_solder_paste_count` 與 `selected_solder_paste`。選取模式為 `contains_center`（包含中心）、`within_tolerance`（容許距離內）或 `none`（無候選），`center_count` 為該階段的候選數量。`obb_ok_count` 仍是全圖達信心門檻的 Solder_Paste 數量，因此外觀通過時也可能大於 1；`obb_retained_detections` 保留所有達門檻的框，包含被忽略的鄰近框。`manifest.inspection_metadata.policy.appearance` 記錄容許比例、中心選取與重疊規則，並納入 `rule_version`。

所有結果須經主控選定才可發布；晚到模型結果不覆蓋已選定的保護結果。每筆工作資料夾的實際影像總數大於 processing.folder_images_num_threshold（預設 60）時，跳過推論並全 23 回傳，原因為 images_exceed_threshold；等於上限仍依 batch_size 分批推論。推論操作上限由 inference.obb.inference_timeout_seconds 設定。權重精度與批次效能需在 AIPC 驗證。
