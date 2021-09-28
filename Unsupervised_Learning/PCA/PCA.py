import numpy as np
import matplotlib.pyplot as plt
import sys


class PCA:

    def __my_copyright(self):
        print("*" * 36)
        print("*", ' ' * 5, 'programmed by KyrieW', ' ' * 5, '*')
        print("*", ' ' * 2, 'you can use it as you like', ' ' * 2, '*')
        print("*", ' ' * 3, "But if you find some bugs", ' ' * 2, '*')
        print('*', "contact me at npgwym@outlook.com", '*')
        print("*" * 36)

    def __init__(self):
        self.__my_copyright()
        self.k = 0
        self.data_x = None
        self.n = None
        self.d = None

    def standardization(self, matrix):
        """
        This method will standardized the data matrix
        :param matrix: the original data each col is a data and each row is a feature
        :return: data after standardization
        """
        # number of row and col
        row, col = matrix.shape[0], matrix.shape[1]

        # generate the original mu, each col of mu is the mean of the mean of each feature
        mu = np.zeros((row, 1))
        # norms(which is the norm of each feature, we use infinity norm here)
        norms = np.zeros((row, 1))
        # standard_matrix is the return matrix
        sta_matrix = np.zeros((row, col))
        # no_mean is the temp matrix we use to record matrix - mean
        no_mean = np.zeros((row, col))

        # get no_mean
        for r in range(row):
            mu[r] = sum(list(matrix[r][:])) / col  # mean for each row
            for c in range(col):
                no_mean = matrix[r][col] - mu[r]
        
        # get norm of each row 
        for r in range(row):
            max_i = -sys.maxsize
            for c in range(col):
                max_i = no_mean[r][c] if no_mean[r][c] > max_i else max_i
            norms[r] = max_i
        
        # divide the norm
        for r in range(row):
            for c in range(col):
                if no_mean[r][c] == 0:
                    sta_matrix[r][c] = no_mean[r][c] / 1
                else:
                    sta_matrix[r][c] = no_mean[r][c] / norms[r]
        
        return sta_matrix, mu, norms

    def variance_matrix(self, s_data_x):
        """
        This method will get the covariance matrix from the standardized data matrix
        :param data_x: the standardized data matrix
        :return: the covariance matrix
        """
        # assume each col is a pic(we take each col of the pic and line them up in one line)
        # doing AT * A is the covariance between pics, A * At is the covariance between pixels
        # you need to think about what do you want before doing this oppration
        return np.cov(s_data_x.T @ s_data_x) # At * A

    def eigen(self, variance_matrix):
        """
        This method will compute the eigenvalue and unit eigenvectors of the variance matrix
        :param variance_matrix: the variance matrix
        :return: eigenvalue matrix, unit eigenvectors
        """
        e_values, e_vectors = np.linalg.eig(variance_matrix)  # the return vectors are unit

        return e_values, e_vectors

    def projection(self, data_x, e_values, e_vectors, k, mu, norms):
        """
        This method will do the projection which means dimension reduction
        :param data_x: data need PCA
        :param e_vectors: the projection bases(eigenvectors)
        :param k: k-main-dimension
        :param mu: mean got in the standardized method
        :param norms: norm got in the standardized method
        :return: data after PCA
        """
        data_copy = np.array(data_x[:])
        e_va = list(e_values)

        # standardizied data_copy
        for r in range(data_copy.shape[0]):
            if norms[r][0] == 0:
                data_copy[r] = (data_copy[r] - mu[r][0]) / 1
            else:
                data_copy[r] = (data_copy[r] - mu[r][0]) / norms[r][0]

        project_matrix = np.zeros((self.n, self.d))
        for i in range(k):
            project_matrix[:, i] = e_vectors[:, i].real  # e_v is real + i * complex
        
        projection_data = data_copy.T @ project_matrix
        return projection_data
        

    def fit(self, data_x):
        """
        This method will receive data from user
        :param data_x: the data need to do PCA
        :return: None
        """
        self.data_x = data_x
        self.n = self.data_x.shape[1]
        self.d = self.data_x.shape[0]

    def transform(self, k):
        """
        This method will do PCA and return the processed data
        :param k: the k main vec
        :return: the data after dimension reduction
        """
        
        sta_matrix, mu, norms = self.standardization(self.data_x)
        variance_matrix = self.variance_matrix(sta_matrix)
        e_va, e_vec = self.eigen(variance_matrix)
        projected_data = self.projection(self.data_x, e_va, e_vec, k, mu, norms)

        return projected_data
        