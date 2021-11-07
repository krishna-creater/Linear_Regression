from os import path
import csv
import numpy as np
from sklearn.metrics import mean_squared_error

def check_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("couldn't find  '"+ predicted_test_Y_file_path+"' file")

def check_format(test_X_file_path, predicted_test_Y_file_path):
    pred_y = []
    with open(predicted_test_Y_file_path,'r') as file:
        reader = csv.reader(file)
        pred_y = np.array(list(reader))
        test_x = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    if pred_y.shape != (len(test_x),1):
        raise Exception("output format is not proper")

def check_mse(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_y = np.genfromtxt(predicted_test_Y_file_path, delimiter=',', dtype = np.float64)
    actual_y = np.genfromtxt(actual_test_Y_file_path, delimiter=',', dtype = np.float64)
    mse = mean_squared_error(actual_y, pred_y)
    return mse

def validate(test_X_file_path, actual_test_Y_file_path):
    predicted_test_Y_file_path = "predicted_test_Y_lr.csv"
    check_file_exits(predicted_test_Y_file_path)
    check_format(test_X_file_path, predicted_test_Y_file_path)
    print(check_mse(actual_test_Y_file_path, predicted_test_Y_file_path))