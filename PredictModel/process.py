import sys
import subprocess

theproc1 = subprocess.Popen([sys.executable, "D:\\庫存健診開發\\code\\PredictModel\\Tuning_tree.py"])
theproc1.communicate()

theproc2 = subprocess.Popen([sys.executable, "D:\\庫存健診開發\\code\\PredictModel\\Tuning_bagging.py"])
theproc2.communicate()

theproc3 = subprocess.Popen([sys.executable, "D:\\庫存健診開發\\code\\PredictModel\\Tuning_boosting.py"])
theproc3.communicate()