import numpy as np
import matplotlib.pyplot as plt


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
        :param matrix: the original data
        :return: data after standardization
        """

        pass

    def variance_matrix(self, s_data_x):
        """
        This method will get the covariance matrix from the standardized data matrix
        :param data_x: the standardized data matrix
        :return: the covariance matrix
        """
        pass

    def eigenvalue(self, variance_matrix):
        """
        This method will compute the eigenvalue of the variance matrix
        :param variance_matrix: the variance matrix
        :return: the eigenvalue
        """
        pass

    def eigenvector(self, variance_matrix, eigenvalue):
        """
        This method will compute the eigenvalue of the variance matrix
        :param variance_matrix: the variance matrix
        :param eigenvalue: the eigenvalue
        :return: eigenvector
        """
        pass

    def k_eigenvector(self, eigenvector):
        """
        select k biggest eigenvalue based, thus we can find the k biggest eigenvectors as our bases
        :param eigenvector: all the eigenvectors
        :return: the k biggest eigenvectors
        """
        pass

    def unit_orthogonal_base(self, eigenvector):
        """
        Processing the bases and turn them into unit orthogonal bases
        :param eigenvector: the bases
        :return: unit orthogonal bases
        """
        pass

    def projection(self, data_x, bases):
        """
        This method will do the projection which means dimension reduction
        :param data_x: data need PCA
        :param bases: the projection bases
        :return: data after PCA
        """
        pass

    def fit(self, data_x):
        """
        This method will receive data from user
        :param data_x: the data need to do PCA
        :return: None
        """
        self.data_x = data_x
        self.n = self.data_x.shape[1]
        self.d = self.data_x.shape[0]

    def transform(self):
        """
        This method will do PCA and return the processed data
        :return: the data after dimension reduction
        """
        pass
