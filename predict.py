import sys
import numpy as np
import csv
from validate import validate
def import_data_and_weights(test_x_file_path, weights_file_path):
    test_x = np.genfromtxt(test_x_file_path, delimiter=',', dtype = np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype = np.float64)
    return test_x, weights

def predict_target_values(test_x, weights):
    test_x = np.insert(test_x, 0, 1, axis = 1)
    # print(test_x)
    # print(weights)
    pred_y = np.dot(test_x, weights)
    return pred_y

def write_to_csv_file(pred_y, predicted_y_file_name):
    pred_y = pred_y.reshape((len(pred_y),1))
    with open(predicted_y_file_name, 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerows(pred_y)
        file.close()

def predict(text_x_file_path):
    text_x, weights = import_data_and_weights(text_x_file_path, "WEIGHTS_FILE.csv")
    pred_y = predict_target_values(text_x, weights)
    write_to_csv_file(pred_y, "predicted_test_Y_lr.csv")


if __name__=="__main__":
    text_X_file_path = sys.argv[1]
    predict(text_X_file_path)
    # validate(text_X_file_path, actual_test_Y_file_path = "train_Y_lr.csv")
