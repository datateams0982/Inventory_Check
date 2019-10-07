# 庫存健診
 
## Crawler
 
* Crawling data from twse
 
## Data Preprocess

* Read and join csv
* Aggregate data
* Extract OHLCV
* Filling missing timestamps

## Clustering

* Denoising stock prices by wavelet
* Build Correlation Matrix
* Clustering
* Select cluster and split data

## Feature Engineering

* Time Feature: Intramonth, week-day, weekly, monthly
* Denoised OHLCV
* Technical Indicators calculated by denoised OHLCV
* Cluster Index
* Return
* Dependent Variable


## Training Preparation

* Feature Scaling
* Transform to (batch, time, feature) shape
* Split into Training and Testing Data


## Model 

* Model
* Hyperparameter Tuning
* Evaluation
* Result Printing

