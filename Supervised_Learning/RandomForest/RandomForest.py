import numpy as np
from Tree import Tree
import random
from scipy import stats


class RandomForest:

    def __init__(self):
        self.__my_copyright()
        self.forest = []
        self.feature_split = []
        self.feature_num = 0
        self.tree_num = 0
        self.votes = []

    def __my_copyright(self):
        print("*" * 36)
        print("*", ' ' * 5, 'programmed by KyrieW', ' ' * 5, '*')
        print("*", ' ' * 2, 'you can use it as you like', ' ' * 2, '*')
        print("*", ' ' * 3, "But if you find some bugs", ' ' * 2, '*')
        print('*', "contact me at npgwym@outlook.com", '*')
        print("*" * 36)

    def _make_data(self, train_x, featureNum, TreeNum):
        """
        This method will choose different combination of featureNum features for each tree
        to learn
        :param train_x: ndarray, train_data
        :param featureNum: int, the number of each tree has to deal with
        :param TreeNum: int, number of trees in the forest
        :return: None
        """
        # generate n index to select feature
        features = train_x.shape[1]
        x_split = []

        # based on the number of trees, generating TreeNum splits
        for i in range(TreeNum):
            feature_index = random.sample(range(0, features-1), featureNum)
            self.feature_split.append(feature_index)

        # select the chosen features for each group of x
        for i in range(train_x.shape[0]):
            temp_x = np.zeros((train_x.shape[0], featureNum))
            for j in range(featureNum):
                temp_x[i][j] = temp_x[i][featureNum[j]]

            x_split.append(temp_x)
        return x_split

    def construct(self, train_x, train_y, depth_limit, featureNum, TreeNum):
        """
        This method will train the forest
        :param train_x: ndarray, training data
        :param train_y: ndarray, training label data
        :param depth_limit: int, the maximum depth of each tree
        :param featureNum: int, number of trees in the forest
        :param TreeNum: int, number of trees in the forest
        :return: None, we will store the trees in self.forest which is a list
        """
        self.feature_num = featureNum
        self.tree_num = TreeNum
        x_splits = self._make_data(train_x,featureNum,TreeNum)
        for i in range(TreeNum):
            tree = Tree(x_splits[i], train_y, depth_limit)
            tree.train()
            self.forest.append(tree)

    def predict(self, predict_x, regression):
        """
        This method will make prediction for the given data point
        :param predict_x: ndarray, the data point
        :param regression: boolean, true the forest will do regression, otherwise it will do classification
        :return: the prediction result
        """
        self.votes = []

        # we need to do the same thing to the input data as we did on the training data
        for i in range(self.tree_num):
            tempx = np.zeros((1, self.feature_num))
            for j in range(self.feature_num):
                for index in self.feature_split[i]:
                    tempx[0][j] = predict_x[0][index]

            vote = self.forest[i].predict(tempx, regression)
            self.votes.append(vote)

        if regression:
            return np.nanmean(self.votes)
        else:
            return stats.mode(self.votes)