# Python-KNN
K-Nearest-Neighbours is a classical supervised learning algorithm which based
on a simple theory -- vote. By calculate the distance between the unclassified data
points X to the train data we already have, find the first nth nearest neighbours. Then we can 
use the data points we find to do the following work.

1. Classification:
    `We already knew the label for the train data, thus we can give a prediction abdou the label of 
    data x based on the mode of the label among the neighbours`
2. Regression:
    `Although we won't acquire and formula or disrtribution information about the data, we still can 
    give our prediction of x by using the mean of its neighbours`

KNN is a model which is easy to understand but its performance on data which contains a huge amount
of data or have a lot of features is not so well. And it need we to preprocess the data.