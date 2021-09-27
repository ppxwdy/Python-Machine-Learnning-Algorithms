import numpy as np
from scipy import stats


class KNN:

    def __init__(self):
        self.__my_copyright()
        self.X = None
        self.Y = None
        self.n = None
        self.features = None
        self.num = None

    def __my_copyright(self):
        print("*" * 36)
        print("*", ' ' * 5, 'programmed by KyrieW', ' ' * 5, '*')
        print("*", ' ' * 2, 'you can use it as you like', ' ' * 2, '*')
        print("*", ' ' * 3, "But if you find some bugs", ' ' * 2, '*')
        print('*', "contact me at npgwym@outlook.com", '*')
        print("*" * 36)

    def fit(self, trainX, trainY, neighbours):
        """
        The first step using the model
        Pass the data to the model
        :param trainX: data points for training
        :param trainY: labels for training
        :param neighbours: The number of neighbours you want to classify the data point
        :return: None
        """
        self.X = trainX
        self.Y = trainY
        self.n = neighbours
        self.features = trainX.shape[1]
        self.num = trainX.shape[0]

    def __vote(self, newX, size):
        """
        This function will compute the distance and find n neighbours for the data
        :param newX: the unclassified data, in the shape of (n, features)
        :param size: the number of input data
        :return: result: the vote result , it is a list of list, which records the distance from the n nearest points to
            the data and their value and label
        """
        result = []
        for x in range(size):
            temp = []
            for i in range(self.num):
                temp.append((np.linalg.norm(newX[x] - self.X[i]), self.Y[i], self.X[i]))
            temp.sort(key=lambda x : x[0])
            # sort by the distance
            temp2 = [temp[j][0] for j in range(self.n)]
            result.append(temp2)
        return result

    def transform_classification(self, newX):
        """
        # This function will classify the data points based on the mode of the vote result of the n nearest points
        :param newX: the data we need to predict
        :return: L: an (n * 1) array, each row is the label of the unclassified data
        """
        size = newX.shape[0]
        res = self.__vote(newX, size)
        L = np.array((size, 1))

        for r in range(len(res)):
            # [0][0] is the mode, [1][0] is the number of the mode appears
            mode = stats.mode(res[r])[0][0]
            L[r, 0] = mode
        return L

    def transform_regression(self, newX):
        """
         # This function will predict the label of x based on the mean value of the n nearest points
        :param newX: the data we need to predict
        :return: Y: an (n * 1) array which records the mean of each points' neighbours
        """
        size = newX.shape[0]
        res = self.__vote(newX, size)
        L = np.array((size, 1))

        for r in range(len(res)):
            mean = np.nanmean([res[i][2] for i in range(size)])
            L[r, 0] = mean
        return L


