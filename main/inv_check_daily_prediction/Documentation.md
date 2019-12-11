# 執行指令

$ python main.py (date)
例: python main.py 2019-12-02
- date 可加可不加，若無則自動預設成今日日期
- date的格式必須為year-month-day

# 如何使用Docker執行

## build
$ docker build --no-cache -t inv_check_daily_prediction -f ./Dockerfile .

## run
$ docker run --rm --name inv_check_daily_prediction --mount type=bind,source=[source_directory]/inv_check_daily_prediction,target=/inv_check_daily_prediction inv_check_daily_prediction

## run with specific date
$ docker run --rm --name inv_check_daily_prediction --mount type=bind,source=[source_directory]/inv_check_daily_prediction,target=/inv_check_daily_prediction inv_check_daily_prediction python /inv_check_daily_prediction/main.py 2019-12-02

# 檔案結構
 
## config: 
- basic_config(基本設定檔): 含資料查詢設定、retry設定、存取資料庫設定、log/result/feature路徑設定、預測模型路徑、舊資料保存天數、特徵工程主要參數設定、應有最少個股數
- feature_config(特徵工程設定檔): 計算平均比例/momentum的天數設定、隨機指標天數設定、RSI時間區間設定、價量關係時間區間設定
- columns_dict(欄位查詢檔): 由特徵工程的主程式(get_technical_indicators)存取，取得需重複進行相同特徵處理的欄位
- feature_dict(特徵明細檔): 紀錄各模型所採用的特徵，在特徵工程結束後可直接取出需要的特徵
- mssqltip_bytes.bin(加密檔): 資料庫密碼加密

## core:
- ALL_STOCK_preprocess_function: 資料存取與清理的函式檔，包含從資料庫撈取需要的資料、填補遺漏天數、結合(產業)指數資料與個股資料
- VWAP_feature_function_rolling_week: 進行特徵處理的函式檔，包含特徵處理的主函式及子函式、存取特徵明細的函式
- Prediction: 進行預測的函式檔，包含將單筆資料送入預測模型並回傳結果、將結果寫入資料庫
- exception_outbound: 將錯誤及程式運行訊息/預測結果傳至telegram

## output:
- log: 存放過去的log file
- prediction: 存放過去的預測結果
- feature: 存放過去特徵資料

## 其他:
- main: 主程式
- requirements: 所需套件及其版本
- Documentation.md: Documentation
 
# 主程式流程

- 讀取基本設定檔
- logging設定
- 檢查特徵檔是否已存在，若有則直接進行預測並寫入資料庫
- 刪除超過保留天數的log file及預測結果
- 從資料庫撈取資料
- 填補遺漏時間
- 結合(產業)指數資料與個股資料
- 讀取欄位查詢檔
- 進行特徵工程
- 讀取特徵明細檔，並取出需要特徵
- 將所需特徵寫入存放資料夾
- 逐筆資料輸入模型預測
- 將預測資料整合後寫入存放資料夾
- 將預測資料寫入資料庫
- 將結果與成功訊息傳至telegram

# 例外處理

- 部分函式將在一段時間過後重新啟動(stock_query, prediction, outbound)
- 部分例外不影響程式運行，將紀錄於log檔後繼續運行剩餘程式(remove previous log and result)
- 影響程式運行的例外，將紀錄於log檔後，將錯誤訊息傳至telegram，程式將接著停止運行

# Debug Guidance 

## 讀取基本設定檔(Configs not in this Directory error): 
- 基本設定檔路徑有誤，確認basic_config.json是否在config資料夾下, 檔名是否正確

## logging設定

## 檢查特徵檔是否已存在(Checking if Feature Engineering is already done):
- 若存在且筆數足夠則直接讀入檔案進行預測，若在預測或寫入資料庫階段發生錯誤，請往下看'逐筆資料輸入模型預測'及'寫入資料夾/資料庫'
- 若否，則執行其餘所有流程

## 刪除超過保留天數的log file及預測結果(Removing previous log and results): 
- 30天前資料不存在或已移除，僅需確認是否已確實移除即可 

## 從資料庫撈取資料(Query Data From ODS.Opendata): 
- Configs not in this Directory: 基本設定檔路徑有誤，確認basic_config.json是否在config資料夾下, 檔名是否正確
- Encoding Document not in this directory: 加密檔路徑有誤，確認mssqltip_bytes.bin是否在config資料夾下, 檔名是否正確
- Data Not Updated: 資料庫資料尚未更新 (暫時設定會在五分鐘後重試，最多重試四次)
- Stock/Index/Industry data length doesn't match: 回傳至python的dataframe長度與query得出之table長度不等
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

## 將所需特徵寫入存放資料夾(Writing Feature to Local):
- 存取問題

## 逐筆資料輸入模型預測(Predicting):
- Configs not in this Directory: 基本設定檔路徑有誤，確認basic_config.json是否在config資料夾下, 檔名是否正確
- 未能連線: 檢查模型伺服器是否開啟、是否有其他連線問題
- 其他錯誤: 檢視traceback並檢視Prediction中的prediction函式

## 將預測資料整合後寫入存放資料夾(Writing to Local): 
- 存取問題

## 將預測資料寫入資料庫(Writing to Database):
- Encoding Document not in this directory: 加密檔路徑有誤，確認mssqltip_bytes.bin是否在config資料夾下, 檔名是否正確
- 其他錯誤: 檢視traceback並檢視Prediction中的write_to_db函式或檢視資料庫連線

## 將訊息傳至telegram(Sending message and result to telegram):
- 若在此段出錯，檢視traceback並檢視exception_outbound中的outbound函式


