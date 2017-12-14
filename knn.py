import matplotlib.pyplot as plt
import numpy as np
import math
import operator
import random

def plot_fig(x,y,file):
    plt.plot(x, y)
    plt.ylabel('Accuracy')
    plt.xlabel('k values')
    plt.savefig(file)

class Knn(object):

    '''
    Findind euclidian distances between 2 instances or vectors
    '''
    def euclidian_distance(self,instance1,instance2):
        return math.sqrt(np.sum(np.square(instance1 - instance2)))
    '''
    Takes n-2D arrays of trainingData
          test_labels per training_labels
          trainingSample to be tested
    '''
    def getneighbours(self,trainingData,training_labels,testSample,k):
        distances = []
        neighbours = []
        for index,trainingInstance in enumerate(trainingData):
            #length = len(testInstance.flatten())
            dist = self.euclidian_distance(trainingInstance,testSample)
            distances.append((training_labels[index],dist))
        distances.sort(key=operator.itemgetter(1))
        for i in range(k):
            neighbours.append(distances[i][0])
        #print neighbours
        return neighbours
    '''
    Takes neighbours and find who occur the most
    '''
    def getmostoccuringlabel(self,neighbours):
        votes = {}
        for neighbour in neighbours:
            if neighbour in votes:
                votes[neighbour] +=1
            else:
                votes[neighbour] =1
        votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return votes[0][0]
    #cross validation
    def shuffle(self,x,y):
        x_y = zip(x,y)
        random.shuffle(x_y)
        x1 ,y1= zip(*x_y)
        return (x1 , y1)

    def get_kth_fold(self,x, y, k, n):
        len_x = x.shape[0]
        split_start = int(float((n - 1) * 1.0 / k) * len(x))
        split_end = int(float((n) * 1.0 / k) * len(x))
        test_x = x[split_start:split_end]
        validation_x = np.concatenate((x[0:split_start], x[split_end:len_x]), axis=0)
        test_y = y[split_start:split_end]
        validation_y = np.concatenate((y[0:split_start], y[split_end: len_x]), axis=0)
        return (test_x, test_y, validation_x, validation_y)
    '''
    Function which runs for all instances in test and gets the accuracy 
    '''
    def prediction(self,train_x, train_y,test_x,test_y,k):
        correct_count = 0
        for x in range(len(test_x)):
            neighbors = self.getneighbours(train_x, train_y, test_x[x], k)
            result = self.getmostoccuringlabel(neighbors)
            if result == test_y[x]:
                correct_count += 1
        accuracy = ((float(correct_count) / len(test_y)) * 100.0)
        return accuracy


if __name__ == "__main__":
    #read Train Label
    train_x_file = "train_data.csv"
    train_x= np.genfromtxt(train_x_file, delimiter=',')
    print (train_x.shape)

    train_y_file = "train_label.csv"
    train_y = np.genfromtxt(train_y_file, delimiter=',')
    print(train_y.shape)

    test_x_file = "test_data.csv"
    test_x = np.genfromtxt(test_x_file, delimiter=',')
    print(test_x.shape)

    test_y_file = "test_label.csv"
    test_y = np.genfromtxt(test_y_file)
    print(test_y.shape)
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
    plot_fig(k_value_acc.keys(),k_value_acc.values(), 'knn.png')