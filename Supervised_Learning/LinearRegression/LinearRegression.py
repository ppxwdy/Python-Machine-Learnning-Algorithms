import numpy as np


class LinearRegression:

    defaultDegree = 5  # the default degree when only one feature
    defaultTimes = 50000
    defaultRate = 0.005

    def __init__(self):
        self.__my_copyright()
        self.xNum = None
        self.features = None
        self.label = None
        self.theta = None
        self.loss = None

    def __my_copyright(self):
        print("*"*36)
        print("*", ' '*5, 'programmed by KyrieW', ' '*5, '*')
        print("*", ' '*2, 'you can use it as you like', ' '*2, '*')
        print("*", ' '*3, "But if you find some bugs", ' '*2, '*')
        print('*', "contact me at npgwym@outlook.com", '*')
        print("*" * 36)

    def get_polynomial_features(self, inData, degree=0):
        """
        get the polynomial features of the data, you can set the degree you
        want when implement the method, the default degree is 5.
        The polynomial features would be the train data of the model
        :param inData: An array x of shape(m,d)
        :param degree: degree of the polynomial
        :return: matrix of shape(m, (degree+1)*features) consisting horizontally concatenated polynomial terms
        """
        # the output matrix, we plus one to degree because we have a
        # zero square term

        real_degree = (degree + 1) * self.features  # for each feature in x, we need to compute them separately
        feature_matrix = np.zeros((self.xNum, real_degree))

        # each column corresponds to a data point, each row correspond to its n square
        for row in range(self.xNum):
            for feature in range(self.features):
                for col in range((degree+1)*feature, real_degree):
                    feature_matrix[row, col] = inData[row][feature] ** col

        return feature_matrix

    def initialize_parameters(self, n):
        """
        generate the initial parameters which means theta
        :param n: n is the number of parameters
        :return: initialTheta : ndarray, the initial parameter theta we generate
        """
        initial_theta = np.zeros((n, 1))
        for i in range(n):
            initial_theta[i, 0] = np.random.randn()
        return initial_theta

    def ms_error(self, X, Theta, Lambda=0):
        """
        This method will calculate the mean square error for the temp parameters
        :param X: the feature matrix
        :param Theta: the parameter matrix
        :param Lambda: the label matrix
        :return:
        """
        error = 0
        for row in range(self.xNum):
            error += ((self.label[row] - X[row, :]@Theta)**2) / 2  # sum up the errors for each sample
        error = error / self.xNum  # 1/N
        norm = np.linalg.norm(Theta)
        return error + Lambda * norm

    def grad(self, featureMatrix, Theta, Y, size=defaultDegree):
        """
        This method compute the grad of the lost function
        :param featureMatrix: the featureMatrix
        :param Theta: the theta matrix
        :param Y: labels
        :param size: the size of the grad matrix, the col num is as same as the number of theta
        :return: float, the grad value
        """
        degree = (size + 1) * self.features

        y = np.zeros((self.xNum, 1))
        x = np.zeros((self.xNum, degree))

        for row in range(self.xNum):
            y[row, :] = Y[row]
            x[row, :] = featureMatrix[row, :]

        grad = (-y.T @ x + Theta.T @ x.T @ x) / self.xNum
        return grad

    def stochastic_descent(self, featureMatrix, Theta, Lambda=0, iterations=defaultTimes, learningRate=defaultRate):
        """
        This method implements the standard descent, which means it will calculate the whole dataset directly
        :param featureMatrix: the feature matrix
        :param Theta: the theta matrix
        :param Lambda: regularization parameter
        :param iterations: the time of iterations
        :param learningRate: the learning rate
        :return: trained theta and the error record for each time(Loss)
        """
        loss = np.zeros((iterations, 1))
        for _ in range(iterations):
            Theta -= learningRate * self.grad(featureMatrix, self.label)  # update theta
            loss[_, 0] = self.ms_error(featureMatrix, Theta, Lambda)

        return Theta, loss

    def batch_descent(self, featureMatrix, Theta, batchSize=20, Lambda=0, iterations=defaultTimes, learningRate=defaultRate):
        """
        This method will do the gradient descent by splitting the data to some batches to accelerate the process
        :param featureMatrix: the feature matrix
        :param Theta: the theta matrix
        :param batchSize: the size of each batch
        :param Lambda: regularization parameter
        :param iterations: the time of iterations
        :param learningRate: the learning rate
        :return:
        """
        degree = (self.defaultDegree + 1) * self.features
        loss = np.zeros((iterations, 1))
        for _ in range(iterations):

            # randomly pick (batch size) data to do the training
            index = np.random.randint(50, size=batchSize)
            y_batch = np.zeros((batchSize, 1))
            x_batch = np.zeros((batchSize, degree))
            for row in range(batchSize):
                y_batch[row, :] = self.label[index[row]]
                x_batch[row, :] = featureMatrix[index[row], :]
            Theta -= learningRate * self.grad(x_batch, Theta, y_batch, degree)
            loss[_, 0] = self.ms_error(featureMatrix, Theta, Lambda)
        return Theta, loss

    def get_theta(self, inData, Label, reg_lambda):
        """
        This method is for the solution of mathematics method of find the solution
        It will find the parameters by calculation
        :param inData: training data
        :param Label: training label
        :param reg_lambda: regularization parameter
        :return:
        """
        dim = (inData.T @ inData).shape[0]
        z = inData.T @ inData + reg_lambda * np.identity(dim)
        return np.linalg.inv(z) @ inData.T @ Label

    def fit_train(self, inData, Label, gd, Lambda=0, batchSize=20,
                  iterations=defaultTimes, rate=defaultRate, degree=defaultDegree):
        """
        This method will calculate the theta by the traditional way, train for n times
        and try to find the local minimum
        :param inData: training data
        :param Label: training label
        :param gd: the parameter control which way this method will use to do gradient descent
        :param Lambda: regularization parameter
        :param batchSize: the size of each batch if we use batch gd
        :param iterations: the time of training
        :param rate: the learning rate
        :param degree: degree of the polynomial
        :return: None
        """
        self.xNum = inData.shape[0]
        self.features = inData.shape[1]
        self.label = Label
        x_train = self.get_polynomial_features(inData, degree)
        theta = self.initialize_parameters((degree+1)*self.features)
        if gd == "normal":
            self.theta, self.loss = self.stochastic_descent(x_train, theta, Lambda, iterations, rate)
        elif gd == "batch":
            self.theta, self.loss = self.batch_descent(x_train, theta, batchSize, Lambda, iterations, rate)

    def fit_compute(self, inData, Label, reg_lambda=0):
        """
        This method get the theta by compute, the default mode is no regularization
        :param inData: training data
        :param Label: training label
        :param reg_lambda: regularization parameter
        :return: None
        """
        self.theta = self.get_theta(inData, Label, reg_lambda)

    def predict(self, data):
        """
        This method will make the prediction
        :param data: the data we need to predict
        :return: the label result matrix
        """
        return data @ self.theta
