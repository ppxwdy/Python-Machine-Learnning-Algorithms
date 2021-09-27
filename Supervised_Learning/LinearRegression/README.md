# Python-Linear-Regression
This model is a simple linear Regression implemented by myself using Python. 
If something is WRONG or could be IMPROVED, please let me know. Thanks a lot.

I offered two kinds of mode to run the program, one is the traditional way, to train the model
by iteration and get the parameters which give us the local minimum. The other one is pure math,
we get the parameters which give us the local minimum by doing matrix calculation. You can try both
and the reason why we have the traditional one is to help us understand how the mechanics work.

The data description:

Because if we use x directly we will need to compute XtX, which could be large and
sometimes non-invertible, I will do the Linear regression with features instead. Linear regression means linear in the features,
so don't worry, we are still doing linear regression.
 
L-R model is Y = X*Theta

xi is the ith training sample, it has m degree, thus it belongs to Rm

X is the training data we get from x, it is a polynomial feature matrix. Each row of X is a training example.

Y := [y1, y2,..., yn]^T which belongs to Rn

Theta is our parameters. What we need from the training is the matrix of theta.

What we have in the beginning is m training samples which all have n dimensions.
``
The average loss of the model is give by:
    Remp(f, X, Y) = 1/N * sum(l(yi, yip)).
    Note that yip is the y we get from the prediction of the model\n
    
    Remp = 1/N * sum( l(yi, f(x*theta)) )
         = 1/N * sum( (yi - f(xi, theta))^2 )
         = 1/N * sum( (yi - xi*theta)^2 )
     which in matrix form is
     L(Theta) = 1/N * (Y - X*Theta)T * (Y- X*Theta) + lambda * ||theta||^2*__*
     
     The equation for the global minimum（least squares method） is given by
     theta = (N*lambda*I + XTX)^-1 * XT * Y
     
     
     
         
    