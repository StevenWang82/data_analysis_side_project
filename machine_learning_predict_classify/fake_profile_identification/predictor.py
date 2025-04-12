import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any # 匯入類型提示

class FakeProfilePredictor:
    def __init__(self, model_path: str = 'models/random_forest_model.pkl',
                 scaler_path: str = 'models/scaler.pkl'):
        """
        初始化預測器，載入模型和 scaler。

        Args:
            model_path (str): 訓練好的模型的路徑 (.pkl)。
            scaler_path (str): 訓練好的 scaler 的路徑 (.pkl)。
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError as e:
            print(f"錯誤：找不到模型或 scaler 檔案：{e}")
            raise
        except Exception as e:
            print(f"載入模型或 scaler 時發生未預期錯誤：{e}")
            raise

        # 定義需要的特徵欄位（確保這個順序與訓練時完全一致！）
        self.required_features = [
            'user_intro', 'user_birthday', 'user_gender', 'user_nickname',
            'user_domicile', 'user_residence', 'user_education', 'user_basic_info_score', 'user_headphoto_yn',
            'user_headphoto_click', 'user_background_yn', 'user_createtime', 'user_linkname',
            'user_numphoto', 'user_headphoto_face', 'user_headphoto_gender_match', 'user_age_diff',
            'user_family', 'user_numfriend', 'user_numtracked', 'user_numtracking',
            'user_numfriend_yn', 'user_numtracked_yn', 'user_numtracking_yn', 'user_workplace',
            'user_phone', 'user_email', 'user_contact_score', 'user_numlikepage',
            'user_numlikepage_yn', 'user_numpost', 'post_share_ratio', 'post_photo_ratio', 'post_act_at',
            'post_act_with', 'post_act_feel', 'post_act_at_with_feel_num', 'post_act_at_with_feel_ratio',
            'post_avg_like', 'post_avg_comment', 'post_avg_share', 'post_recent_yn'
        ]
        self._required_features_set = set(self.required_features)

    def _prepare_dataframe(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        將輸入數據統一轉換為 Pandas DataFrame，並檢查與排序特徵。
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy() # 使用副本避免修改原始傳入的 DataFrame
        elif isinstance(data, list):
            if not data:
                 raise ValueError("輸入數據列表為空")
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise TypeError("輸入數據必須是 Pandas DataFrame、字典或字典列表")

        # 檢查必要欄位
        current_columns_set = set(df.columns)
        missing_features = self._required_features_set - current_columns_set
        if missing_features:
            raise ValueError(f"輸入數據缺少必要特徵: {missing_features}")

        # 檢查是否有不需要的額外欄位 (可選，但有助於保持輸入乾淨)
        # extra_features = current_columns_set - self._required_features_set
        # if extra_features:
        #     print(f"警告：輸入數據包含額外欄位: {extra_features}，這些欄位將被忽略。")

        # 確保欄位順序正確，並只選取需要的欄位
        try:
            # 使用 reindex 確保順序，同時處理欄位選擇
            df_ordered = df.reindex(columns=self.required_features)
            # 檢查 reindex 後是否引入了 NaN (如果原始 df 缺少某欄位但檢查被繞過)
            # 理論上，前面的 missing_features 檢查可以防止這種情況
            # if df_ordered.isnull().any().any():
            #      raise ValueError("選取特徵後發現缺失值 (NaN)，請檢查原始數據。")
        except Exception as e:
             raise ValueError(f"根據 required_features 重排/選取欄位時發生錯誤: {e}")

        return df_ordered

    def _validate_numeric_types(self, df: pd.DataFrame):
        """
        檢查 DataFrame 中的所有欄位是否均為數值類型。
        """
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            # 找出非數值欄位及其類型
            non_numeric_info = {col: df[col].dtype for col in non_numeric_cols}
            raise TypeError(
                f"Scaler/模型預期所有特徵均為數值型數據，但檢測到以下非數值型欄位: "
                f"{non_numeric_info}. "
                f"請在調用 predict/predict_proba 前，確保將這些欄位轉換為適當的數值類型 (例如 int 或 float)。"
            )

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> List[int]:
        """
        預測輸入資料是否為假帳號。

        Args:
            data: 包含所需特徵的 Pandas DataFrame、單個字典或字典列表。

        Returns:
            List[int]: 預測結果列表。1 通常表示真實帳號，0 表示假帳號 (取決於模型訓練)。
        """
        try:
            # 1. 準備 DataFrame (選擇、排序欄位)
            df_prepared = self._prepare_dataframe(data)

            # 2. 驗證數據類型是否均為數值型
            self._validate_numeric_types(df_prepared) # <--- 加入檢查點

            # 3. 特徵縮放
            X_scaled = self.scaler.transform(df_prepared)

            # 4. 進行預測
            predictions = self.model.predict(X_scaled)

            # 5. 將 NumPy array 轉換為 Python list
            return predictions.tolist()

        except (ValueError, TypeError) as e: # 捕捉準備/驗證階段的錯誤
             print(f"數據準備或類型驗證失敗: {e}")
             raise # 重新拋出，讓調用者知道出錯了
        except Exception as e: # 捕捉縮放或預測階段的錯誤
            print(f"特徵縮放或模型預測時發生未預期錯誤: {e}")
            # 可能需要記錄更詳細的信息
            raise # 重新拋出

    def predict_proba(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> List[List[float]]:
        """
        預測輸入資料為各個類別的機率。

        Args:
            data: 包含所需特徵的 Pandas DataFrame、單個字典或字典列表。

        Returns:
            List[List[float]]: 預測機率列表。
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("載入的模型不支援 predict_proba 方法。")

        try:
            # 1. 準備 DataFrame
            df_prepared = self._prepare_dataframe(data)

            # 2. 驗證數據類型是否均為數值型
            self._validate_numeric_types(df_prepared) # <--- 加入檢查點

            # 3. 特徵縮放
            X_scaled = self.scaler.transform(df_prepared)

            # 4. 進行機率預測
            probabilities = self.model.predict_proba(X_scaled)

            # 5. 將 NumPy array 轉換為 Python list
            return probabilities.tolist()

        except (ValueError, TypeError) as e:
             print(f"數據準備或類型驗證失敗: {e}")
             raise
        except Exception as e:
            print(f"特徵縮放或模型預測機率時發生未預期錯誤: {e}")
            raise

# --- 使用範例 ---
if __name__ == '__main__':
    # 假設你已經有模型和 scaler 在 'models/' 目錄下
    try:
        predictor = FakeProfilePredictor()

        # 範例 1: 使用單個字典預測
        single_data_dict = {
            'user_intro': 1, 'user_birthday': 1990, 'user_gender': 1, 'user_nickname': 1,
            'user_domicile': 1, 'user_residence': 1, 'user_education': 3, 'user_basic_info_score': 80, 'user_headphoto_yn': 1,
            'user_headphoto_click': 1, 'user_background_yn': 1, 'user_createtime': 2015, 'user_linkname': 0,
            'user_numphoto': 15, 'user_headphoto_face': 1, 'user_headphoto_gender_match': 1, 'user_age_diff': 2,
            'user_family': 0, 'user_numfriend': 200, 'user_numtracked': 50, 'user_numtracking': 30,
            'user_numfriend_yn': 1, 'user_numtracked_yn': 1, 'user_numtracking_yn': 1, 'user_workplace': 1,
            'user_phone': 0, 'user_email': 1, 'user_contact_score': 70, 'user_numlikepage': 10,
            'user_numlikepage_yn': 1, 'user_numpost': 50, 'post_share_ratio': 0.1, 'post_photo_ratio': 0.8, 'post_act_at': 1,
            'post_act_with': 1, 'post_act_feel': 0, 'post_act_at_with_feel_num': 2, 'post_act_at_with_feel_ratio': 0.04,
            'post_avg_like': 20, 'post_avg_comment': 3, 'post_avg_share': 1, 'post_recent_yn': 1
            # ... 確保所有 required_features 都在這裡，值僅為示例 ...
        }
        prediction = predictor.predict(single_data_dict)
        probabilities = predictor.predict_proba(single_data_dict)
        print(f"單筆字典預測結果: {prediction}") # 應該輸出類似 [1] 或 [0]
        print(f"單筆字典預測機率: {probabilities}") # 應該輸出類似 [[0.1, 0.9]]

        # 範例 2: 使用 DataFrame 預測 (假設 df_test 是你的測試 DataFrame)
        # df_test = pd.read_csv('your_test_data.csv')
        # # 確保 df_test 包含所有 required_features
        # if all(col in df_test.columns for col in predictor.required_features):
        #      predictions_batch = predictor.predict(df_test)
        #      probabilities_batch = predictor.predict_proba(df_test)
        #      print(f"\n批次 DataFrame 預測結果 (前5筆): {predictions_batch[:5]}")
        #      print(f"批次 DataFrame 預測機率 (前5筆): {probabilities_batch[:5]}")
        # else:
        #      print("\n測試 DataFrame 缺少必要欄位，跳過批次預測示例。")

        # 範例 3: 使用字典列表預測
        list_of_dicts = [
            single_data_dict, # 第一筆資料
            single_data_dict.copy() # 第二筆資料 (這裡用同一筆複製做示例)
            # ... 可以加入更多字典 ...
        ]
        # 你可以修改第二筆字典中的一些值來觀察差異
        list_of_dicts[1]['user_numfriend'] = 5
        list_of_dicts[1]['user_numpost'] = 0

        predictions_list = predictor.predict(list_of_dicts)
        probabilities_list = predictor.predict_proba(list_of_dicts)
        print(f"\n字典列表預測結果: {predictions_list}") # 應該輸出類似 [1, 0]
        print(f"字典列表預測機率: {probabilities_list}") # 應該輸出類似 [[0.1, 0.9], [0.8, 0.2]]


    except FileNotFoundError:
        print("錯誤：無法運行範例，因為找不到模型/scaler 檔案。")
    except ValueError as e:
        print(f"運行範例時發生錯誤: {e}")
    except Exception as e:
         print(f"運行範例時發生未預期錯誤: {e}")