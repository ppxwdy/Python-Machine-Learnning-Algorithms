import numpy as np
from TreeNode import TreeNode
from scipy import stats


class Tree:

    def __init__(self, train_x, train_y, limit):
        self.train_x = train_x
        self.train_y = train_y
        self.limit = limit
        self.root = TreeNode(0, self.train_x, self.train_y, self.limit)

    def train(self):
        self.root.train()

    def predict(self, predict_x, regression):
        """
        This method will make prediction based on the training-result
        Input:
          predict_x: ndarray, the data we need to give prediction
          root: TreeNode, the root node of the tree
          regression: a boolean variable shows we are doing classification(False) or regression(True)
        Output:
          regression: mean of vote
          classification: mode of vote
        """

        vote = []
        self.root.predict(predict_x, vote, regression)
        if regression:
            return np.nanmean(regression)
        else:
            return stats.mode(vote)[0][0]
