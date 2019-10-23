# 庫存健診
 
## Crawler
 
* Crawling data from twse
 
## Data Preprocess

* Read and join csv
* Filter out non-trading stocks
* Combine with capital reduction data
* Filling missing timestamps

## Clustering

* Compute time weighted mean of true range, capital and total as feature of clustering
* log normalization, separate to two class by total
* Clustering
* Select cluster and split data

## Feature Engineering

* Remove Reduction data and separate data.
* Denoised OHLCV
* Technical Indicators calculated by denoised OHLCV
* Cluster Index


## Training Preparation

* Feature Scaling
* Transform to (batch, time, feature) shape
* Split into Training and Testing Data


## Model 

* Model: CNN_tree, CNN_bagging, CNN_boosting
* Hyperparameter Tuning
* Evaluation
* Result Printing

