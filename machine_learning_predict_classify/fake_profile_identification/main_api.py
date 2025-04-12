# main_api.py
import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ValidationError # 用於定義請求和響應的資料結構
from typing import List, Dict, Any, Optional

# --- 從你的 predictor.py 匯入類別 ---
try:
    from predictor import FakeProfilePredictor
except ImportError:
    logging.error("錯誤：無法從 predictor.py 匯入 FakeProfilePredictor。")
    logging.error("請確保 predictor.py 檔案存在於正確的位置。")
    # 在 API 啟動時就失敗，而不是等到請求進來
    raise RuntimeError("無法啟動 API：找不到 Predictor 類別")

# --- 配置 Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 初始化 FastAPI 應用 ---
app = FastAPI(
    title="假帳號個人檔案預測 API (Fake Profile Predictor API)",
    description="提供使用者個人檔案特徵，預測其是否為假帳號。",
    version="1.0.0"
)

# --- 全局 Predictor 實例 ---
# 在應用啟動時只載入一次模型，提高效率
try:
    predictor = FakeProfilePredictor()
    logging.info("Predictor 初始化成功，模型已載入。")
except Exception as e:
    logging.error(f"初始化 Predictor 時發生嚴重錯誤: {e}", exc_info=True)
    # 如果模型載入失敗，API 應該無法正常工作
    raise RuntimeError(f"無法啟動 API：Predictor 初始化失敗 - {e}")

# --- 定義請求的資料模型 (使用 Pydantic) ---
# 這個模型需要包含 predictor 需要的所有欄位，並且類型要正確
# 這樣 FastAPI 可以自動驗證傳入的 JSON 資料
class ProfileFeatures(BaseModel):
    user_intro: int
    user_birthday: int
    user_gender: int
    user_nickname: int
    user_domicile: int
    user_residence: int
    user_education: int
    user_basic_info_score: float # 或 int，根據你的數據
    user_headphoto_yn: int
    user_headphoto_click: int
    user_background_yn: int
    user_createtime: int
    user_linkname: int
    user_numphoto: int
    user_headphoto_face: int
    user_headphoto_gender_match: int
    user_age_diff: float # 或 int
    user_family: int
    user_numfriend: int
    user_numtracked: int
    user_numtracking: int
    user_numfriend_yn: int
    user_numtracked_yn: int
    user_numtracking_yn: int
    user_workplace: int
    user_phone: int
    user_email: int
    user_contact_score: float # 或 int
    user_numlikepage: int
    user_numlikepage_yn: int
    user_numpost: int
    post_share_ratio: float
    post_photo_ratio: float
    post_act_at: int
    post_act_with: int
    post_act_feel: int
    post_act_at_with_feel_num: int
    post_act_at_with_feel_ratio: float
    post_avg_like: float
    post_avg_comment: float
    post_avg_share: float
    post_recent_yn: int

    # 添加一個 Config 來自訂 Pydantic 的行為 (例如，允許額外字段但忽略它們)
    # class Config:
    #     extra = 'ignore' # 如果傳入了不在模型中的字段，將其忽略而不是報錯

    # 添加一個範例，會顯示在 API 文件中
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'user_intro': 1, 'user_birthday': 1990, 'user_gender': 1, 'user_nickname': 1,
                    'user_domicile': 1, 'user_residence': 1, 'user_education': 3, 'user_basic_info_score': 80.0, 'user_headphoto_yn': 1,
                    'user_headphoto_click': 1, 'user_background_yn': 1, 'user_createtime': 2015, 'user_linkname': 0,
                    'user_numphoto': 15, 'user_headphoto_face': 1, 'user_headphoto_gender_match': 1, 'user_age_diff': 2.0,
                    'user_family': 0, 'user_numfriend': 200, 'user_numtracked': 50, 'user_numtracking': 30,
                    'user_numfriend_yn': 1, 'user_numtracked_yn': 1, 'user_numtracking_yn': 1, 'user_workplace': 1,
                    'user_phone': 0, 'user_email': 1, 'user_contact_score': 70.0, 'user_numlikepage': 10,
                    'user_numlikepage_yn': 1, 'user_numpost': 50, 'post_share_ratio': 0.1, 'post_photo_ratio': 0.8, 'post_act_at': 1,
                    'post_act_with': 1, 'post_act_feel': 0, 'post_act_at_with_feel_num': 2, 'post_act_at_with_feel_ratio': 0.04,
                    'post_avg_like': 20.0, 'post_avg_comment': 3.0, 'post_avg_share': 1.0, 'post_recent_yn': 1
                }
            ]
        }
    }


# --- 定義響應的資料模型 ---
class PredictionResponse(BaseModel):
    predictions: List[int] # 預測結果列表 (0 或 1)
    api_model_version: str = "1.0" # (可選) 添加模型版本信息

class ProbabilityResponse(BaseModel):
    probabilities: List[List[float]] # 預測機率列表
    api_model_version: str = "1.0"

# --- 定義 API 端點 (Endpoints) ---

@app.get("/", summary="API 健康檢查 (Health Check)")
async def read_root():
    """
    訪問根目錄，檢查 API 是否正在運行。
    """
    return {"message": "Fake Profile Predictor API is running!"}

# 使用 POST 方法，因為客戶端需要發送資料給 API
# request body 會被自動解析並驗證為 List[ProfileFeatures]
@app.post("/predict/",
          response_model=PredictionResponse,
          summary="預測是否為假帳號 (0 或 1)")
async def predict_fake_profile(profiles: List[ProfileFeatures]):
    """
    接收一個或多個使用者個人檔案的特徵列表，返回對應的預測結果 (0=假帳號, 1=真實帳號)。

    - **profiles**: 一個包含一個或多個使用者資料字典的列表。每個字典需要符合 `ProfileFeatures` 的結構。
    """
    if not profiles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="輸入的 profiles 列表不能為空。"
        )

    try:
        # 將 Pydantic 模型列表轉換為 Predictor 期望的字典列表
        # 使用 .model_dump() (Pydantic v2) 或 .dict() (Pydantic v1)
        data_dicts = [profile.model_dump() for profile in profiles]

        # 調用 predictor 進行預測
        logging.info(f"收到 {len(data_dicts)} 筆資料進行預測...")
        results = predictor.predict(data_dicts)
        logging.info(f"預測完成，返回 {len(results)} 個結果。")

        # 返回結果
        return PredictionResponse(predictions=results)

    except (ValueError, TypeError) as data_err:
        # 捕捉由 predictor 引發的數據準備/驗證錯誤
        logging.error(f"預測時發生數據錯誤: {data_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # 400 表示客戶端請求有問題
            detail=f"輸入資料處理失敗: {data_err}"
        )
    except NotImplementedError as ni_err:
         logging.error(f"調用了不支援的方法: {ni_err}", exc_info=True)
         raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(ni_err)
        )
    except Exception as e:
        # 捕捉其他所有未預期的錯誤
        logging.error(f"預測過程中發生未預期服務器錯誤: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # 500 表示伺服器內部錯誤
            detail=f"伺服器內部預測錯誤: {e}"
        )

@app.post("/predict_proba/",
          response_model=ProbabilityResponse,
          summary="預測為各類別的機率")
async def predict_profile_probability(profiles: List[ProfileFeatures]):
    """
    接收一個或多個使用者個人檔案的特徵列表，返回對應的預測機率列表。
    每個子列表通常包含 `[假帳號的機率, 真實帳號的機率]`。

    - **profiles**: 一個包含一個或多個使用者資料字典的列表。每個字典需要符合 `ProfileFeatures` 的結構。
    """
    if not profiles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="輸入的 profiles 列表不能為空。"
        )

    try:
        # 轉換為字典列表
        data_dicts = [profile.model_dump() for profile in profiles]

        # 調用 predictor 進行機率預測
        logging.info(f"收到 {len(data_dicts)} 筆資料進行機率預測...")
        probabilities = predictor.predict_proba(data_dicts)
        logging.info(f"機率預測完成，返回 {len(probabilities)} 個結果。")

        # 返回結果
        return ProbabilityResponse(probabilities=probabilities)

    except (ValueError, TypeError) as data_err:
        logging.error(f"預測機率時發生數據錯誤: {data_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"輸入資料處理失敗: {data_err}"
        )
    except NotImplementedError as ni_err:
        # 如果模型不支持 predict_proba
         logging.error(f"調用了不支援的方法: {ni_err}", exc_info=True)
         raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(ni_err)
        )
    except Exception as e:
        logging.error(f"預測機率過程中發生未預期服務器錯誤: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"伺服器內部預測機率錯誤: {e}"
        )

# --- (可選) 運行指令提示 ---
# 這部分只是為了方便，實際運行是通過 uvicorn 命令
if __name__ == "__main__":
    import uvicorn
    print("="*50)
    print(" FastAPI 應用程式已準備就緒 ".center(50, "="))
    print("="*50)
    print("若要啟動伺服器，請在終端機中執行：")
    print("\n    uvicorn main_api:app --reload --host 0.0.0.0 --port 8000\n")
    print("參數說明:")
    print("  main_api: Python 檔案名稱 (main_api.py)")
    print("  app: 在 main_api.py 中創建的 FastAPI() 實例")
    print("  --reload: 開發時使用，當程式碼變更時自動重啟伺服器")
    print("  --host 0.0.0.0: 允許來自區域網路內其他機器的訪問 (預設只允許本機)")
    print("  --port 8000: 指定伺服器監聽的埠號 (預設 8000)")
    print("\n啟動後，可在瀏覽器訪問以下網址：")
    print("  API 主頁 (Health Check): http://127.0.0.1:8000/")
    print("  互動式 API 文件 (Swagger UI): http://127.0.0.1:8000/docs")
    print("  替代 API 文件 (ReDoc): http://127.0.0.1:8000/redoc")
    print("="*50)
    # 可以直接運行，但不推薦用於生產環境，且不支援 --reload
    # uvicorn.run(app, host="127.0.0.1", port=8000)


    # 在terminal 輸入 👇
    # uvicorn main_api:app --reload