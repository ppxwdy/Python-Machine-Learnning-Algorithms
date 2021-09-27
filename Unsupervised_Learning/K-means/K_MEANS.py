import sys
import numpy as np


class Clustering:

    def __init__(self):
        self.__my_copyright()
        self.dataNum = None
        self.features = None
        self.trainx = None
        self.n = None

    def __my_copyright(self):
        print("*" * 36)
        print("*", ' ' * 5, 'programmed by KyrieW', ' ' * 5, '*')
        print("*", ' ' * 2, 'you can use it as you like', ' ' * 2, '*')
        print("*", ' ' * 3, "But if you find some bugs", ' ' * 2, '*')
        print('*', "contact me at npgwym@outlook.com", '*')
        print("*" * 36)

    def fit(self, trainx, n):
        """
        Call this function to pass the basic information to the model
        :param trainx: train data points
        :param n: the number of clusters you want
        :return: None
        """
        self.dataNum = trainx.shape[0]
        self.features = trainx.shape[1]
        self.trainx = trainx
        self.n = n

    def initialize(self, n):
        """
        generalized the initial parameters
        :param n: number of centroids, each has d dimensions
        :return: None
        """

        centroids = np.zeros((n, self.features))

        # randomly pick the first centroid
        index = np.random.randint(0, self.dataNum)

        centroids[0, :] = self.trainx[index, :]

        # Select the rest centroids which a
        # We select the centroids by calculate the distance between the
        # new centroid and the centroids we have already
        for i in range(1, n):  # the nth centroid
            max_dis = 0
            temp_c = np.array((1, self.features))
            for p in self.trainx:  # go through all the data in the dataset
                dis = 0
                for c in centroids:
                    dis += np.linalg.norm(c - p)
                    if max_dis < dis:
                        max_dis = dis
                        temp_c[0, :] = c[0, :]
            centroids[i, :] = temp_c[0, :]

        return centroids

    def assign(self, C):
        """
        This function will assign each of the data point to its nearest
        centroid.
        :param C: the centroids matrix
        :return: the assignment relation of each data in the trainX to its temp centroid
             has the same shape as trainX, each of its row is a centroid from centroids
            which shows the centroid for the data in train who has the index .
        """
        assigment = np.zeros((self.dataNum, self.features))
        for i in range(self.dataNum):
            min_dis = sys.maxsize
            temp_c = -1
            for c in C:
                dis = np.linalg.norm(self.trainx[i, :] - c)
                if dis < min_dis:
                    min_dis = dis
                    temp_c = c
            assigment[i, :] = temp_c
        return assigment

    def update(self, C, assignment, n):
        """
        This function will calculate the mean of each cluster formed by temp centroids
        Then it will define a new centroid for each cluster by using the mean of the data points
        :param C: ndarry, the temp centroids
        :param assignment: ndarry, the temp assignment for data points
        :param n: number of data points
        :return: new centroids
        """
        new_c = np.zeros(C.shape)

        for i in range(n):
            cluster = np.zeros((1, self.features))
            num = 0
            for j in range(self.dataNum):
                if np.array_equal(C[i, :], assignment[j, :]):
                    cluster += self.trainx[j, :]
                    num += 1
            new_c[i, :] = cluster / num

        return new_c

    def iteration(self, time, n):
        """
        this function will iterate the assign and update process for time times
        :param time: int, how many times do you want to train
        :param n: number of data points
        :return: The final centroids and assignment
        """
        centroids = self.initialize(n)
        assignment = self.assign(centroids)

        for t in range(time):
            centroids = self.update(centroids, assignment, n)
            assignment = self.assign(centroids)

        return centroids, assignment

    def transform(self, time, n):
        """
        allocate the data points to the cluster it belongs
        :param time: int, how many times do you want to train
        :param n: number of data points
        :return: A list of arrays, an array is a cluster
        """
        centroids, assignment = self.iteration(time, n)
        clusters = []
        for c in centroids:
            temp = []
            for j in range(assignment.shape[0]):
                if np.array_equal(assignment[j, :], c):
                    temp.append(self.trainx[j, :])
            clusters.append(np.asarray(temp))

        return clusters
