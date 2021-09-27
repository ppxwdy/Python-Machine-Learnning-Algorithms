import numpy as np
from scipy import stats

# think about how to deal with the feature
# By index or by name or transfer before using


class TreeNode:

    def __init__(self, layer, X, Y, limit, kind=None):
        """
        construct method
        attribution:
          layer: an integer shows the layer of this node
          X: an ndarray shows the set data point
          Y: an ndarray shows the label of the data
          limit: an integer shows the max_depth of this tree
          kind: an specific type of variable, shows which kind of the last layer's feature this node represents
        """
        self.layer = layer
        self.next_ = []
        self.X = X
        self.Y = Y
        self.num = X.shape[0]
        self.feature = None
        self.featureNum = X.shape[1]
        self.limit = limit
        self.kind = kind

    def _category_entropy(self):
        """
        This method will calculate the category entropy
        Output:
            info_label (info(D))
        """
        # use a hash table to select distinct labels
        self.labels = set(self.Y)

        # count the number of distinct labels
        self.count = dict()
        for y in self.Y:
            if y not in self.count:
                self.count[y] = 1
            else:
                self.count[y] += 1
        info_label = 0
        for label in self.labels:
            info_label -= (self.count[label] / self.num) * np.log2(np.asarray([self.count[label] / self.num]))[0]
        return info_label

    def _attribution_entropy(self):
        """
        This method will calculate the attribution entropy
        Output:
          info_a: a dict record all the entropy of each attribution
        """
        # label-feature : lf =dict[label] , lf[label1][feature] = number
        self.lf = dict()
        info_a = dict()

        # Count the number of different values of each attribute under the corresponding label
        # label
        for label in self.labels:
            for i in range(self.num):
                # attribution
                if self.Y[i] == label:
                    for feature in range(self.featureNum):
                        k = self.X[feature]
                        # numOfDifferent, kind = self.countCategory()
                        if label not in self.lf:
                            self.lf[label] = dict()
                        if feature not in self.lf[label]:
                            self.lf[label][feature] = 0
                        if k not in self.lf[label][feature]:
                            self.lf[label][feature][k] = 0
                        self.lf[label][feature][k] += 1

        # calculate the info of each feature
        for f in range(self.featureNum):
            info_a[f] = 0
            temp = 0
            # the part when label = this_label
            for label in self.labels:
                this_label = 0
                # log前的系数 num(label) / all
                A = self.count[label] / self.num
                # turn the dict of this feature into list
                # so we can calculate the value for each format of this attribution
                for k in list(self.lf[label][f]):
                    this_label -= (self.lf[label][f][k] / self.count[label]) * np.log2(self.lf[label][f][k] / self.count[label])
                temp -= A * this_label
            info_a[f] += temp

        return info_a

    def _gain(self):
        """
        The method will calculate the gain
        Output:
          A dict records the gain for each attribution
        """
        info_label = self._category_entropy()
        info_a = self._attribution_entropy()
        gain_a = dict()
        for f in range(self.featureNum):
            gain_a[f] = info_label - info_a[f]
        return gain_a

    def _categorical_information_measurement(self):
        """
        This method will calculate the categorical Information Measurement
        Output:
          h_a which is a dict about h of each attribution
        """
        h_a = dict()

        # we need to find the amount of each kind of each attribution(feature)
        for f in range(self.featureNum):
            amount_k, kinds = self._count_kind(f)
            for k in kinds:
                count = 0
                for i in range(self.num):
                    if self.X[i][f] == k:
                        count += 1
                h_a[f] -= count/self.num * np.log2(count/self.num)
        return h_a

    def _info_gain_ratio(self):
        """
        This method will calculate the IGR
        Output:
          igr: a list of tuple, which is the feature and its igr
        """
        igr = []
        h_a = self._categorical_information_measurement()
        gain_a = self._gain()
        for f in range(self.featureNum):
            igr.append((f, gain_a[f]/h_a[f]))
        return igr

    def choose_feature(self):
        """
        This method will decide which feature we will use to split the data
        We use C4.5 algorithm in this function which means we will select the feature
        which has the maximum information entropy
        Output:
          The feature we will use in the next layer
        """
        # sort the igr list by the igr value fo each feature
        igr = self._info_gain_ratio().sort(key=lambda x: x[1])

        # the biggest igr will be place at the end of the list
        # thus we can find the feature for next layer
        return igr[-1][0]

    def _count_kind(self, feature):
        """
        This method will count the number of different format of this feature
        This will decide how many child node this node will have
        Output:
          the number of different format of this feature
          """
        kinds = set()
        for i in range(self.num):
            kinds.add(self.X[i][feature])
        return len(kinds), kinds

    def _split(self, feature):
        """
        This method will based on the difference of the given feature to split the dataset
        Output:
          we will have a list of child_node of this node which is divided based on the feature
        """
        num_o_k, kinds = self._count_kind(feature)

        for k in kinds:
            next_x = []
            next_y = []
            for x in range(self.num):
                if self.X[x][self.layer] == k:
                    next_x.append(self.X[x])
                    next_y.append(self.Y[x])
            child_node = TreeNode(np.asarray(next_x), np.asarray(next_y), self.limit, k)
            self.next_.append(child_node)

    def _check_pure(self):
        """
        This method will check whether the node is pure
        Output:
          A boolean value, if it is pure, then return True, otherwise return False
        """
        if len(set(self.Y)) == 1:
            return True
        else:
            return False

    def train(self):
        if self.layer == self.limit:
            return
        if self._check_pure():
            return
        # this method will decide which feature will be used in this layer
        self.feature = self.choose_feature()
        self._split(self.feature)
        for n in self.next_:
            n.train()

    def predict(self, predict_x, vote, regression):
        """
        This method will make prediction based on the training-result
        Input:
          predict_x: ndarray, the data we need to give prediction
          vote: the list to record all the prediction of all the leaf nodes
          regression: a boolean variable shows we are doing classification(False) or regression(True)
        """
        # if this is the leaf node, it will give its prediction by find the mode of the vote result
        if self._check_pure():
            # classification
            if not regression:
                mode = stats.mode(self.Y)[0][0]
                vote.append(mode)
                return
            # regression
            else:
                mean = np.nanmean(self.Y)
                vote.append(mean)
                return

        # check the value for the given feature
        kind = predict_x[self.feature]
        for node in self.next_:
            if kind == node.kind:
                return node.predict(predict_x, regression)

