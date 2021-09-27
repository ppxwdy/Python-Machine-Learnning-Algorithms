# Python-PCA

We want to keep information while dimension reduction as much as possible, find the eigenvalue of 
the covariance matrix can help us find the biggest variance between each data after projection( Because
 we want the projection data to be as dispersed as possible, cuz if there is overlapping, some info will
 disappear), thus we can get the target of our dimension reduction: make a n-d data to k-d by finding k 
 unit orthogonal bases, the data after transformation will have 0 covariance and k biggest variance.
 
Procedure:
```
1\ make the original data a d*n matrix (n data, each has d-dimension) X
2\ Standardization
3\ compute the covariance matrix C
4\ compute the eigenvalue and eigenvector of C
5\ rank the K biggest eigenvector from large to small as the transform Matrix T_M
6\ get the new data X_k = T_m * X
```

## Standardization
**_ATTENTION:
!!!!!!!!
Do standardization to both training data and test data but they have to be done separately. Otherwise
the info of test data will be blended into training data, which make the test looks better but we
are actually cheating._**

A data pre-processing method which means make the data have 0 mean and 1 variance

For PCA, we are actually doing singular value decomposition which doesn't need to do this
at all. But if the value of some feature is way more larger than other value, then we could ignore
the influence of the smaller value which would result in losing information, thus we need to 
do this in machine learning.


## Covariance Matrix
```
Assume we have n random variables, X = (X1, ..., Xn)^T
The covariance between two variables is 
            cov[Xi, Xj] = E[(Xi - E[Xi]] * E[(Xj - E[Xj]] 
Then the n*n covariance matrix is

cov = [ cov(1, 1) cov[1, 2], ... , cov[1, n]
                        .
                        .
                        .
        cov(n, 1) cov(n, 2), ... , cov[n, n] ]
```
To explain why we have to do this before applying PCA, we need to know the meaning of covariance.
```
1\ The covariance shows the linear correlation between two variables
2\ Each element in the matrix shows the linear correlation between any two variables
3\ The matrix shows the linear correlation between any two variables inside a group of random variblse
```
