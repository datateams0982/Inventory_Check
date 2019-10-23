from tqdm import tqdm_notebook as tqdm
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
import os
from datetime import datetime, timedelta, date


class TrainTestSplit:

    def __init__(self, df_path, cluster_num, )