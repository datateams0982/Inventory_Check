# 執行指令

$ python main.py (date)
例: python main.py 2019-12-02
- date 可加可不加，若無則自動預設成今日日期
- date的格式必須為year-month-day

# 檔案結構
 
## config: 
- basic_config(基本設定檔): 含資料查詢設定、retry設定、存取資料庫設定、log/result路徑設定、預測模型路徑、舊資料保存天數、特徵工程主要參數設定
- feature_config(特徵工程設定檔): 計算平均比例/momentum的天數設定、隨機指標天數設定、RSI時間區間設定、價量關係時間區間設定
- columns_dict(欄位查詢檔): 由特徵工程的主程式(get_technical_indicators)存取，取得需重複進行相同特徵處理的欄位
- feature_dict(特徵明細檔): 紀錄各模型所採用的特徵，在特徵工程結束後可直接取出需要的特徵
- mssqltip_bytes.bin(加密檔): 資料庫密碼加密

## core:
- ALL_STOCK_preprocess_function: 資料存取與清理的函式檔，包含從資料庫撈取需要的資料、填補遺漏天數、結合(產業)指數資料與個股資料
- VWAP_feature_function_rolling_week: 進行特徵處理的函式檔，包含特徵處理的主函式及子函示、存取特徵明細的函式
- Prediction: 進行預測的函式檔，包含將單筆資料送入預測模型並回傳結果、將結果寫入資料庫

## output:
- log: 存放過去的log file
- prediction: 存放過去的預測結果

## 其他:
- main: 主程式
- requirements: 所需套件及其版本
- Documentation.md: Documentation
 
# 主程式流程

- 讀取基本設定檔
- logging設定
- 刪除超過保留天數的log file及預測結果
- 從資料庫撈取資料
- 填補遺漏時間
- 結合(產業)指數資料與個股資料
- 讀取欄位查詢檔
- 進行特徵工程
- 讀取特徵明細檔，並取出需要特徵
- 逐筆資料輸入模型預測
- 將預測資料整合後寫入存放資料夾
- 將預測資料寫入資料庫

# Debug Guidance 

## 讀取基本設定檔(Configs not in this Directory error): 
- 基本設定檔路徑有誤，確認basic_config.json是否在config資料夾下, 檔名是否正確

## logging設定

## 刪除超過保留天數的log file及預測結果(Removing log and results): 
- 30天前資料不存在或已移除，僅需確認是否已確實移除即可 

## 從資料庫撈取資料(Query Data From ODS.Opendata): 
- Configs not in this Directory: 基本設定檔路徑有誤，確認basic_config.json是否在config資料夾下, 檔名是否正確
- Encoding Document not in this directory: 加密檔路徑有誤，確認mssqltip_bytes.bin是否在config資料夾下, 檔名是否正確
- Data Not Updated: 資料庫資料尚未更新 (暫時設定會在五分鐘後重試，最多重試四次)
- StockIIndex/Industry data length doesn't match: 回傳至python的dataframe長度與query得出之table長度不等
- 其他錯誤: 檢視traceback並檢視ALL_STOCK_preprocess_function中的stock_query/send_query函式或資料庫連線問題

## 填補遺漏時間(Filling Missing Time): 
- 若在此段出錯，檢視traceback並檢視ALL_STOCK_preprocess_function中的FillMissingTime函式

## 結合(產業)指數資料與個股資料(Merging Stock Data with Index Data): 
- 若在此段出錯，檢視traceback並檢視ALL_STOCK_preprocess_function中的merge_index函式

## 讀取欄位查詢檔(Reading column dict): 
- 欄位查詢檔路徑有誤，確認columns_dict.json是否在config資料夾下, 檔名是否正確

## 進行特徵工程(Feature Engineering): 
- Configs not in this Directory: 特徵工程設定檔路徑有誤，確認feature_config.json是否在config資料夾下, 檔名是否正確
- 其他錯誤: 檢視traceback並檢視VWAP_feature_function_rolling_week中的get_features函式

## 讀取特徵明細檔，並取出需要特徵(Reading feature list):
- Feature Dict not in this Directory: 特徵明細檔路徑有誤，確認feature_dict.json是否在config資料夾下, 檔名是否正確
- 其他錯誤: 檢視traceback並檢視VWAP_feature_function_rolling_week中的read_feature_lists函式

## 逐筆資料輸入模型預測(Predicting):
- Configs not in this Directory: 基本設定檔路徑有誤，確認basic_config.json是否在config資料夾下, 檔名是否正確
- 未能連線: 檢查模型伺服器是否開啟、是否有其他連線問題
- 其他錯誤: 檢視traceback並檢視Prediction中的prediction函式

## 將預測資料整合後寫入存放資料夾(Writing to Local): 
- 存取問題

## 將預測資料寫入資料庫(Writing to Database):
- Encoding Document not in this directory: 加密檔路徑有誤，確認mssqltip_bytes.bin是否在config資料夾下, 檔名是否正確
- 其他錯誤: 檢視traceback並檢視Prediction中的write_to_db函式或檢視資料庫連線


