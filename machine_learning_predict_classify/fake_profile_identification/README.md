# Facebook 假帳號偵測專案 (Machine Learning)

本專案旨在利用機器學習技術，分析 Facebook 用戶的個人資料與互動行為，以偵測並識別潛在的假帳號（非真實用戶）。
這裡分享的專案，主要用於學習機器學習模型的建立、訓練與評估的通用過程，並提供一個可部署的預測器與 API 介面。

## 專案目標

*   透過分析用戶的靜態資料（如個人檔案資訊）與動態資料（如貼文、按讚、留言等互動行為），建立有效的假帳號偵測模型。
*   比較不同機器學習演算法（SVM、隨機森林、Stacking 融合模型）在偵測任務上的效能。
*   提供一個可部署的模型預測器以及 API 介面。

## 檔案結構說明

```
.
├── data/ (檔案涉及隱私資訊，未上傳至 GitHub)                    # 存放原始資料、預處理後的資料
│   ├── 1130_Final_資料.xlsx
│   ├── missing_data_imputation_file.csv # 遺漏值遞補後的資料
│   ├── 一般_final.csv          # 主要使用的原始資料之一
│   ├── 卡提諾_final_with_predictions.csv
│   └── 卡提諾_final.csv
├── image_output/             # 存放探索性資料分析 (EDA) 的視覺化圖檔
│   ├── EDA_pic_distribution_output_no_missing.docx # EDA 圖檔彙整文件
│   └── ... (各特徵分佈圖)
├── models/                   # 存放訓練好的模型與預處理工具
│   ├── random_forest_model.pkl # 儲存的隨機森林模型
│   └── scaler.pkl              # 儲存的 MinMaxScaler
├── output/                   # 存放模型評估結果，如混淆矩陣圖
│   ├── Confusion matrix of RandomForecast.png
│   ├── Confusion matrix of stacking model.png
│   └── Confusion matrix of SVM.png
├── evaluate_predictor.ipynb  # 用於評估 predictor.py 效能的 Notebook
├── evaluate_predictor.py     # 評估 predictor.py 效能的腳本
├── fake-profiles-identification-ml-technique-simple.ipynb # 主要的分析與模型訓練 Notebook
├── main_api.py               # 提供模型預測功能的 FastAPI 應用程式
├── predictor.py              # 載入模型並進行預測的腳本
├── requirement.txt           # 專案所需的 Python 套件列表
└── README.md                 # 本說明文件
```

## 主要流程與發現 (基於 `fake-profiles-identification-ml-technique-simple.ipynb`)

1.  **資料載入與預處理**:
    *   從 `./data/一般_final.csv` 讀取資料。
    *   使用基於 `class` 分組的機率分佈方法，對資料中的遺漏值進行遞補，處理後的資料存於 `data/missing_data_imputation_file.csv`。

2.  **探索性資料分析 (EDA)**:
    *   針對各項特徵，繪製直方圖比較真實用戶 (`class=1`) 與非真實用戶 (`class=0`) 的分佈差異。
    *   視覺化結果儲存於 `image_output/` 目錄下，並彙整至 Word 文件。

3.  **模型訓練與評估**:
    *   將資料分割為訓練集與測試集。
    *   使用 `MinMaxScaler` 對特徵進行縮放。
    *   訓練並評估了三種模型：
        *   支援向量機 (Support Vector Machine, SVM) - 使用多項式核心 (polynomial kernel)。
        *   隨機森林 (Random Forest)。
        *   Stacking 融合模型 (以 SVM 和隨機森林為基底模型，隨機森林為最終模型)。
    *   計算並輸出各模型的分類報告 (Precision, Recall, F1-score) 與混淆矩陣。混淆矩陣圖儲存於 `output/` 目錄。

4.  **模型儲存**:
    *   將訓練完成的隨機森林模型 (`random_forest_model.pkl`) 與特徵縮放器 (`scaler.pkl`) 儲存至 `models/` 目錄，以供後續 `predictor.py` 和 `main_api.py` 使用。

5.  **主要發現**:
    *   **隨機森林表現優異**: 在此任務中，隨機森林模型展現出高達 99% 的準確率，且訓練與預測速度快。
    *   **互動特徵的重要性**: 相較於容易偽造的靜態個人資料（如年齡、生日），用戶的社群互動行為特徵對於辨識假帳號更具鑑別力。
    *   **關鍵辨識特徵**:
        *   `user_age_diff` (年齡資訊差異): 真實用戶的年齡通常與照片相符，假帳號則差異較大。
        *   `post_avg_comment` (貼文平均留言數): 真實用戶互動較頻繁，假帳號互動較少。
        *   `post_act_at_with_feel_ratio` (貼文打卡與情感標記佔比): 真實用戶更傾向於使用這些功能。

## 環境設定與使用

1.  **安裝依賴套件**:
    開啟終端機 (Terminal) 或命令提示字元 (Command Prompt)，移動到專案根目錄下，執行以下指令安裝所需套件：
    ```bash
    pip install -r requirement.txt
    ```

2.  **執行分析 Notebook**:
    使用 Jupyter Notebook 或 Jupyter Lab 開啟 `fake-profiles-identification-ml-technique-simple.ipynb`，即可查看完整的資料處理、分析與模型訓練流程。

3.  **使用預測器**:
    可以透過執行 `predictor.py` 來載入已儲存的模型進行預測（需要提供輸入資料）。相關的評估可參考 `evaluate_predictor.py` 或 `evaluate_predictor.ipynb`。

4.  **啟動 API 服務**:
    執行以下指令啟動 FastAPI 應用程式，提供模型預測的 API 接口：
    ```bash
    uvicorn main_api:app --reload
    ```
    (請確保已安裝 `uvicorn`)
    啟動後，可透過瀏覽器或 API 測試工具訪問 API (預設路徑通常是 `http://127.0.0.1:8000`)。

## 依賴套件

本專案主要依賴以下 Python 套件 (詳見 `requirement.txt`):

### 數據處理
*   `pandas`: 資料處理與分析
*   `numpy`: 數值計算

### 視覺化
*   `matplotlib`: 基礎資料視覺化
*   `seaborn`: 基於 Matplotlib 的進階統計視覺化

### API 服務
*   `fastapi`: 建立 API 服務
*   `uvicorn`: ASGI 伺服器，用於運行 FastAPI
*   `pydantic`: FastAPI 用於資料驗證

### 機器學習
*   `scikit-learn`: 機器學習函式庫 (資料分割、預處理、模型訓練、評估)
*   `joblib`: 儲存與載入 Python 物件 (模型、Scaler)

### 文件處理
*   `python-docx`: 讀寫 Word 文件 (用於 EDA 報告)

## 注意事項

*   檔案資料 (`.xlsx`, `.csv`) 已經過去識別化處理，使用時請注意隱私保護。
*   模型效能可能因資料集或特徵工程的調整而有所變化。
