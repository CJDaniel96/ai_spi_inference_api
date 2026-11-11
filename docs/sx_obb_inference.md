# SX OBB 判定

單一主控使用常駐模型子程序執行七類 OBB：Solder_Paste、Stain、Missing、Insufficient、Slump、Foreign_object、Bridge。

模型先使用 inference.obb.conf 過濾候選框，再使用 class_thresholds 的各類別門檻過濾。門檻必須不小於模型 conf，避免已在模型端丟棄的框無法恢復。通過外觀必須恰好一個合格 Solder_Paste，且沒有任何合格 NG 框。

固定判定順序為外觀、volume、area、height、shift。數值規則直接讀取 CSV 每列的門檻；沒有 component 特例或另一組 config 門檻。

| 項目 | 量測 | 門檻 | 比較方式 |
|---|---|---|---|
| volume | insp_vol | vol_l_ng、vol_h_ng | 將兩端正規化為 min/max，包含邊界 |
| area | insp_area | area_l_ng、area_h_ng | 將兩端正規化為 min/max，包含邊界 |
| height | insp_hei | hei_l_ng、hei_h_ng | 將兩端正規化為 min/max，包含邊界 |
| shift X | insp_x | shift_x_ng | abs(X) 小於等於非負門檻 |
| shift Y | insp_y | shift_y_ng | abs(Y) 小於等於非負門檻 |

量測與門檻必須是有限數值。必要資料缺漏、模型失敗、模型結果不完整或超時時，整份工作走全 23 保護；正常推論也可能因品質判定全部 NG，兩者透過 manifest.outcome 區分。

processed CSV 保留逐列判定原因、門檻、原始與保留框資訊，以及模型／規則版本。SPI 回傳 CSV 只更新原始 is_pass（22 通過，23 不通過），保留欄位、編碼、列序與原始值。

所有結果須經主控選定才可發布；晚到模型結果不覆蓋已選定的保護結果。每筆工作資料夾的實際影像總數大於 processing.folder_images_num_threshold（預設 60）時，跳過推論並全 23 回傳，原因為 images_exceed_threshold；等於上限仍依 batch_size 分批推論。推論操作上限由 inference.obb.inference_timeout_seconds 設定。權重精度與批次效能需在 AIPC 驗證。
