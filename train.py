import numpy as np
import csv
def import_data():
    a = np.genfromtxt('train_X_lr.csv',delimiter=',',dtype=np.float64, skip_header=1)
    print(a.shape)
    b = np.genfromtxt('train_Y_lr.csv',delimiter=',',dtype=np.float64)
    print(b.shape)
    return a,b

def compute_gradient_of_cost_function(x,y,w):
    y_pred = np.dot(x,w)
    diff = y_pred-y
    dw = (1/(len(x)))*(np.dot(x.T,diff))
    return dw
def compute_cost(x,y,w):
    y_pred = np.dot(x,w)
    mse = np.sum(np.square(y_pred-y))
    cost_value = mse/(2*len(x))
    return cost_value
def optimize_weights_using_gradient_descent(x,y,w,num_iterations, learning_rate):
    previous_iter_cost = 0
    iter_no = 0
    while True:
        iter_no +=1
        dw = compute_gradient_of_cost_function(x,y,w)
        w = w-(learning_rate*dw)
        cost = compute_cost(x,y,w)
        # print(cost,iter_no)
        # if iter_no%100000 ==0:
        #     print(iter_no, cost)
        if abs(previous_iter_cost-cost) < 0.000001:
            print(iter_no,cost)
            break
        # if(iter_no==1000):
        #     break;

        previous_iter_cost = cost
    return w
def train_model(x,y):
    x = np.insert(x,0,1,axis = 1)
    y = y.reshape((len(x),1))
    w = np.zeros((x.shape[1],1))
    w = optimize_weights_using_gradient_descent(x,y,w,1000000,0.0002)
    return w
def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    x,y = import_data()
    weights = train_model(x,y)
    save_model(weights, "WEIGHTS_FILE.csv")
