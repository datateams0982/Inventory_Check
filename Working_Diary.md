# 2019/10/07

* Cluster after denoised
* Feature Engineering by denoised OHLCV 
* CNN on regression (2 conv1D)/classification (6 conv1D), both failed

# 2019/10/08

* Change number of classes to 2.
* Model performance increased when taking away all time features in classification problem (for CNN, LSTM)
* Remove maxpooling layer from CNN (Denoised)
* Tried XGboost, find that previous 5 days are the most important.
* Performance heterogeneity between different stocks in the same cluster(3X~7X)
* Average performance(accuracy) : 60%~65%
* Suggestions: Check the momentum of the cluster, check intra-cluster detail, expecially those clusters with small size.

# 2019/10/09

* Check cluster momentum: Can't last for long time
* New cluster method: Base on volume volatility, volume size, price volatility (time weighted mean)
* Two-stage Cluster: Kmedians, apply cluster algo again on those with large size

# 2019/10/14

* Cluster Method: Base on volume size, price volatility (Remove outliers)
* CNN on cluster 2 and 4
* check performance on middle point and others

# 2019/10/16

* Cluster Method: Base on total size, price volatility, capital (log transformation and standardization)
* largest 60% and others separated
* Update 2008/06/19 data
* Preprocess fundamental and capital reduction data
* Add reduction period indicator to raw data

# 2019/10/17

* Feature Engineering Complete: Kurtosis, skew, price volume trend
* Model experiment: CNN, CNN_randomforest, CNN_XGboost
* Building classes for fine tuning

# 2019/10/21

* Experiment: log transform on OHLCV (perform worse than original)
* Building class for Basic CNN, CNN_tree, CNN_bagging, CNN_boosting 
* Build class for hyperparameter tuning (random search)

# 2019/10/22

* Solving memory problem for CNN-XGBoost (two script, slow, use random forest only?)
* Debug

# 2019/10/23

* Design automated hyperparameter tuning (finished)
* Building classes for clustering, feature engineering, training preparation

# 2019/10/24

* Parameter Tuning

# 2019/10/25

* Parameter Tuning
* First version of model complete
* Better tuning method?
