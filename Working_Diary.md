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
