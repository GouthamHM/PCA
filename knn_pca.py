from pca import  PCA
import  numpy as np
import operator
from knn import Knn

import matplotlib.pyplot as plt

def plot_fig(x,y,file):
    plt.plot(x, y)
    plt.ylabel('Accuracy')
    plt.xlabel('k values')
    plt.savefig(file)

if __name__ == "__main__":
    #read Train Label
    train_x_file = "./data/train_data.csv"
    train_x= np.genfromtxt(train_x_file, delimiter=',')
    print (train_x.shape)

    train_y_file = "./data/train_label.csv"
    train_y = np.genfromtxt(train_y_file, delimiter=',')
    print(train_y.shape)

    test_x_file = "./data/test_data.csv"
    test_x = np.genfromtxt(test_x_file, delimiter=',')
    print(test_x.shape)

    test_y_file = "./data/test_label.csv"
    test_y = np.genfromtxt(test_y_file)
    print(test_y.shape)

    #addition PCA here
    pca = PCA(166,50)
    train_x = pca.getPC(train_x)
    test_x = pca.projectData(test_x)  # standardization will be performed inside the function
    print (train_x.shape)
    print (test_x.shape)
    knn = Knn()
    print type(train_x)
    new_train_x, new_train_y = knn.shuffle(train_x,train_y)
    new_train_x = np.array(new_train_x)
    new_train_y = np.array(new_train_y)
    k_values = [2*l+1 for l in range(0,9)]
    k_value_acc = {i:0.0 for i in k_values}
    for k in k_values:
        sum_accuracy = 0
        folds = 5
        for i in range(1,folds+1):
            train_x1, train_y1, test_x1,test_y1 =  knn.get_kth_fold(new_train_x,new_train_y,folds,i)
            sum_accuracy += knn.prediction(train_x1,train_y1,test_x1,test_y1,k)
        print sum_accuracy/folds
        k_value_acc[k] = sum_accuracy/folds
    print k_value_acc
    sorted_k = sorted(k_value_acc.iteritems(), key=operator.itemgetter(1), reverse=True)
    best_k = sorted_k[0][0]
    accuracy_of_best_k = knn.prediction(train_x, train_y, test_x, test_y, best_k)
    print "best K: "
    print best_k
    print "accuracy"
    print accuracy_of_best_k
    plot_fig(k_value_acc.keys(),k_value_acc.values(),'./captures/knn_pca.png')
