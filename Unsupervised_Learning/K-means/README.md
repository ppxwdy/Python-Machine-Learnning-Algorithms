# Python-K-MEANS
K-means is a unsupervised learning algorithm which will automatically assign the data points 
to k clusters as you required. It could be a foundation of other algorithms such as GMM, 
the algorithms which need to initialize the data points into different group but do not need the actual
label of them.

K-means is an algorithm of k clustering. It is based on the mean value of each cluster.
It is based on the following process:
```
Initilization: 
    Randomly pick k centroids(Usually we want they could be well separated, cuz it will affect the result)
    Then we assign the data points to these centroids based on the distacne of the point and the centroid.
Iteration:
    Caculate the mean value of each cluster, then we will get a location of a point which loacte at the mean postion
of the cluster. It may not be a point belongs to the dataset we have, but this is fine, we just use it to deine the cluster,
it doesn't mean anything.
    We assign the data to the new centroids based on the distance again, then we will have a new distribution.
    The disrtibution won't vary much after a reasonable big number of times of iteration. That is the result.
Assesment:
    We can plot the result after iteration and sometimes we can find some strange distribution or their might be 
some empty clusters, it might be we set the number k wrong, thus we need to re-do the whole process.
```
Here is a example of the plot before and after being processed.

<img width="394" alt="pre" src="https://user-images.githubusercontent.com/58164010/132935559-486630a2-c170-4a7a-a4b3-8e67de95cab2.png">
<img width="384" alt="after" src="https://user-images.githubusercontent.com/58164010/132935561-c239218f-c36b-409d-b73a-8cc6bd8124ba.png">


