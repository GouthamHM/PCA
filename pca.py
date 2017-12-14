import numpy as np

class PCA:
    def __init__(self,N,d):
        self.N = N
        self.d = d

    '''
    Project the data to PCA
    '''
    def project(self,data,eig_val,eig_vec):
        eigen_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
        eigen_pairs.sort()
        eigen_pairs.reverse()
        w = np.hstack((eigen_pairs[i][1].reshape(self.N,1)) for i in range(0,self.d))
        return data.dot(w)

    '''
    Helper function for standardizing data
    '''
    def standardize(self,data):
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        return_data = (data-mean)/std
        return return_data
    '''
    To roject Train Samples
    '''
    def projectData(self,data):
        std_data = self.standardize(data)
        return self.project(std_data,self.eig_val,self.eig_vec)

    '''
    Fetches PCA
    '''
    def getPC(self,samples):
        standarized_data = self.standardize(samples)
        # Find the covariance matrix
        cov = np.cov(standarized_data.T)
        # Find the eigen value and eigen Vector
        eig_val,eig_vec = np.linalg.eig(cov)
        self.eig_val = eig_val
        self.eig_vec = eig_vec
        return self.project(standarized_data,eig_val,eig_vec)
