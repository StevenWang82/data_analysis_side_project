# 時間序列季節性分析 (`seansonality_analysis.ipynb`)

本 Jupyter Notebook 旨在對特定主題「網路聲量數據」，執行時間序列分解與季節性分析，目的是找出時間序列中潛在的模式，例如趨勢（Trend）、季節性（Seasonality）和殘差（Residuals）。

## 工作流程

1.  **環境設定：** 匯入必要的 Python 函式庫，包含 `pandas`、`matplotlib` 和 `statsmodels`。（注意：第一個用於連接 Google Drive 的儲存格是 Google Colab 環境專用）。
2.  **資料載入：**
    *   從指定的 CSV 檔案（例如：`綠色飲食/綠色飲食聲量趨勢.csv`）讀取時間序列資料。
    *   將 'Time' 欄位解析為日期時間物件，並設定為索引（Index）。
    *   將主要的資料欄位重新命名為 'buzz'。
    *   處理潛在的零值，將所有數值加 1（這是乘法分解模型的必要步驟）。
    *   繪製原始時間序列資料圖。
3.  **離群值偵測與處理：**
    *   說明：網路聲量為民眾討論特定議題的文章數，特定議題或是抽獎活動容易大量累積聲量，然而卻不一定能反映實際的趨勢。因此離群值偵測會在此段進行處理。
    *   使用 Z 分數（z-score）方法來識別顯著偏離平均值的資料點。
    *   使用者可以定義 Z 分數的閾值 (`z_determination`) 以及用於取代離群值的移動平均窗口大小 (`window_size`)。
    *   偵測到的離群值會儲存至 `極端值數據_z-score.csv`。
    *   視覺化呈現原始資料、偵測到的離群值、移動平均線和閾值的圖表會儲存為 `極端值檢測_z-score.png`。
    *   透過迭代方式，將離群值替換為移動平均值。
    *   繪製處理離群值後的資料圖。
4.  **時間重採樣 (Time Resampling)：**
    *   將清理過的的時間序列資料，以不同的時間間隔（例如：每日 'D'、每週 'W'、每雙週 '2W'）重新取樣，計算每個區間內的數值總和。
5.  **季節性分解 (Seasonal Decomposition)：**
    *   針對每個重新取樣的時間序列：
        *   應用 `statsmodels.tsa.seasonal_decompose` 將序列分解為趨勢、季節性和殘差成分。
        *   測試加法（'additive'）和乘法（'multiplicative'）兩種模型。
        *   測試與時間間隔相關的不同季節性週期（例如：半年和年度週期）。
        *   根據最低的平均絕對殘差值，決定最佳的模型與週期組合。
        *   將分解圖儲存為 `<time_interval>時間序列拆解圖.png`。
        *   將分解後的成分（趨勢、季節性、殘差）儲存至 `<time_interval>時間序列拆解.csv`。

## 必要套件 (Requirements)

所需的 Python 函式庫列於 `requirements.txt` 檔案中：
```
pandas
matplotlib
statsmodels
```
您可能也需要 `os` 函式庫，這是 Python 的標準函式庫。

## 使用方式

1.  確保已安裝必要的函式庫 (`pip install -r requirements.txt`)。
2.  將您的輸入 CSV 檔案（需包含 'Time' 欄位和一個資料欄位）放置於子目錄中（例如：`綠色飲食/`）。
3.  修改儲存格 #2 中的 `url_base` 和 `file_name` 變數，使其指向您的輸入檔案。
4.  如有需要，可在儲存格 #3 中調整離群值偵測的參數，例如 `z_determination` 和 `window_size`。
5.  如果您想分析不同的時間間隔或季節性週期，請修改儲存格 #4 中的 `units` 和 `periods_to_tes_list`。
6.  依序執行 Notebook 中的所有儲存格。

## 輸出檔案 (Outputs)

此 Notebook 會在指定的 `url_base` 目錄中產生數個輸出檔案：

*   `極端值數據_z-score.csv`: 偵測到的離群值列表。
*   `極端值檢測_z-score.png`: 顯示離群值的圖表。
*   `<time_interval>時間序列拆解圖.png`: 每個時間間隔的分解圖。
*   `<time_interval>時間序列拆解.csv`: 每個時間間隔的分解資料。
