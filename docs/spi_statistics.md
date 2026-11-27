# SPI 統計

統計只讀取 `backup_output_root/YYYY-MM-DD/job_id/ai_result/run.json` 及同行 `manifest.json`，不依賴 SQLite、state_root 或 history_root，也不修改輸入或觸發回傳。

```bat
spi_statistics.bat --output reports
spi_statistics.bat --date 2026-11-09 --output reports
```

也可執行 `python -m app.tools.spi_statistics --config config/ai_server.json --output reports`。

- `jobs.csv`：每個時間戳資料夾最近一批處理一列，包含 job_id、run_id、狀態、階段時間、耗時、實際回傳結果數量、影像備份與 MiNiFi 匯出狀態。
- `deliveries.csv`：每份 CSV 每次回傳嘗試一列，包含起訖時間、耗時、狀態、錯誤、雜湊、來源完成標記與改名時間。
- `summary.csv`：執行次數、成功／失敗／部分成功／不明次數、回傳 PASS／FAIL 數量、平均回傳耗時與待匯出數。

同一份 CSV 取消 `_DONE` 後重新處理，仍有新的 run_id，但會更新同一個 ai_result/run.json。統計只計算目前報告中的最近一批，不累積先前重跑歷史。PASS／FAIL 只計算已確認回傳的 CSV。`--date` 依首次發現的台灣日期篩選，與來源資料夾日期可不同。

結果 manifest 依報告同行路徑讀取，整份備份搬移後仍可統計；先前使用 run_id 子目錄的報告與沒有 ai_result/run.json 的舊備份不會混入新版統計。報告內容損壞時工具會報錯，請保留原檔核對。

## 量測超限分布統計（每 5 單位）

使用 `app/tools/measurement_statistics.py` 統計「影像外觀 AI 判 OK，但 CSV 的體積、面積、高度或位移超限」的影像數。此工具只依賴 Python 標準函式庫，可以單獨複製到有 Python 3.12 的電腦執行。

Windows 在專案目錄執行：

```bat
measurement_statistics.bat "D:\SPI\backup" --start-date 2026-11-01 --end-date 2026-11-24 --output "D:\SPI\reports\measurement_statistics.csv"
```

直接執行 Python，或將這支檔案單獨複製到其他電腦：

```bash
python app/tools/measurement_statistics.py /path/to/backup --start-date 2026-11-01 --end-date 2026-11-24 --output /path/to/reports/measurement_statistics.csv
```

日期依 manifest 的 `created_at` 轉為台灣時間（UTC+8），**包含起日與迄日**；不是影像檔名或來源資料夾日期。省略起日或迄日，該端便不限制。backup 可指定整個備份根目錄、其中一天的資料夾，或單一 `manifest.json`。輸出檔案必須放在輸入備份範圍之外。

### 資料來源與計數方式

1. 遞迴尋找 `manifest.json`，依日期選取 `outcome=normal` 的紀錄；`fallback` 保護性 NG 不納入量測統計。
2. 依 `csv_results[].processed_csv` 讀取相對路徑；若檔案已由 MiNiFi 搬移，改讀同行 `exported` 中的同名 CSV。整份 backup 搬移後仍適用。
3. 檢查 CSV 列數、`is_pass` 是否與 manifest 的 `result_codes` 一致，並讀取 `ai_rule_checks` 中 `yolo_obb.passed=true` 確認影像 AI 判 OK。最終 `is_pass=23` 可以是量測 NG，不會因此排除。
4. 預設 `--mode all`。新備份從 `ai_rule_checks` 讀取實際套用的體積與面積門檻及量測值。舊備份依該列原始 CSV 量測與門檻重算體積、面積、高度及位移。同一影像可在多個項目各計一次，因此各項合計可能大於 NG 影像數。

如果只需要原始判定記錄的**第一個 NG 項目**，加上 `--mode first-failure`。此模式只分析 `ai_failed_rule` 指定的項目。新備份的判定順序為體積、面積，舊備份另含高度、位移。

| 舊備份項目 | 量測欄位 | 門檻欄位 | 超限量計算 |
|---|---|---|---|
| 體積 volume | `insp_vol` | `vol_l_ng`、`vol_h_ng` | 低於下限：下限 − 量測值；高於上限：量測值 − 上限 |
| 面積 area | `insp_area` | `area_l_ng`、`area_h_ng` | 同上 |
| 高度 height | `insp_hei` | `hei_l_ng`、`hei_h_ng` | 同上 |
| 位移 shift | `insp_x`、`insp_y` | `shift_x_ng`、`shift_y_ng` | `max(abs(X) − X門檻, abs(Y) − Y門檻)` |

新舊規則的上下限都包含邊界。舊備份的體積、面積、高度以兩個 CSV 門檻的較小值及較大值為上下限。百分比量測的超限量以百分點計，例如上限 150、量測 156，超限量為 **6 個百分點**。舊備份的位移以 **mm** 計，負位移先取絕對值。X/Y 都超限時只計一張，以較大超限量分組，歸入 `high_ng_count`。

### 分組與小數

預設組距 `--bin-width 5`，以**超過門檻多少**分組，不是以原始量測值分組。使用 Decimal 保留小數，不先四捨五入：

| CSV 的 bin_range | 判斷方式 | 整數數據對應範例 |
|---|---|---|
| `(0,5]` | 0 < 超限量 ≤ 5 | 1～5；超限 0 代表 OK，不計 NG |
| `(5,10]` | 5 < 超限量 ≤ 10 | 6～10 |
| `(10,15]` | 10 < 超限量 ≤ 15 | 11～15 |

例如高度上限 150，量測 154.8 計入 `(0,5]`，155 計入 `(0,5]`，155.1 或 156 計入 `(5,10]`；下限 50、量測 44，也計入超限量 6 的 `(5,10]`，但屬於 `low_ng_count`。

位移若也用 5 mm 分組，通常會集中在第一組；需要更細的位移分布，可另外設定組距，例如 **0.005 mm（5 µm）**：

```bat
measurement_statistics.bat "D:\SPI\backup" --start-date 2026-11-01 --end-date 2026-11-24 --bin-width 5 --shift-bin-width 0.005 --output "D:\SPI\reports\measurement_statistics.csv" --details "D:\SPI\reports\measurement_details.csv"
```

### 輸出 CSV 與製圖

`--output` 指定的主檔每個項目、每個超限區間一列，包含：

| 欄位 | 用途 |
|---|---|
| `metric`、`metric_zh` | 英文項目與中文名稱 |
| `unit` | 原始量測單位：`%` 或 `mm` |
| `bin_index` | 從 1 開始的區間順序，供數值排序 |
| `bin_range` | 例如 `(0,5]`、`(5,10]`，可作為圖表 X 軸 |
| `excess_gt`、`excess_le` | 超限區間下界（不含）與上界（包含） |
| `low_ng_count` | 低於下限的影像數 |
| `high_ng_count` | 高於上限的影像數；位移表示絕對位移超限 |
| `ng_count` | 該區間總數，可作為圖表 Y 軸 |

各項從第一組輸出到該項最大的有資料區間，中間沒有資料的區間填 0；整項沒有超限時，保留第一組 0。Excel 可篩選 `metric_zh=高度`，按 `bin_index` 排序，以 `bin_range` 為 X 軸、`ng_count` 為 Y 軸建立柱狀圖；也可將 `low_ng_count`、`high_ng_count` 做堆疊圖。

另會產生同目錄的 `<主檔名>_errors.csv`，記錄 manifest、CSV 名稱、資料列號與錯誤。可選的 `--details` 檔案則每張影像、每個超限項目一列，保留影像名稱、量測值、門檻、超限量與分組；位移另附 X/Y 值與各軸門檻。CSV 全部使用 UTF-8 BOM，方便 Excel 開啟，每次執行覆寫同名檔。

終端機會列出符合日期的 manifest 數、跳過的 fallback 數、有效／異常 CSV 數、影像外觀 OK／NG 列數、異常列數、量測 NG 影像數（`numeric_ng_images`）及各項超限次數合計（`numeric_ng_events`）。缺少 CSV、列數或結果碼不一致時整份 CSV 排除；缺少 AI 判定記錄、數值無效、量測與最終判定不一致時整列排除，避免部分計入。只要有錯誤，或沒有可分析的影像 AI OK 資料，exit code 為 1；仍會輸出 CSV，請先核對錯誤及涵蓋數量。

manifest 本身不含逐列量測數據，必須保留它對應的 processed／exported CSV。工具只讀備份，不執行推論或修改回傳。每份找到的 manifest 各計一次；相同備份若複製多份放在搜尋範圍內會重複計數，請指定單一備份範圍。現行系統覆寫的舊批次無法還原。
