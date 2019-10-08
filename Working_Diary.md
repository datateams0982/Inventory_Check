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
