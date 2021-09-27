# Python-Random-Forest
## Random Forest:
```
ATTENTION: Random Forest could do classification and Regression, but I used C4.5 algorithm to choose
             feature which made the model can only do classification. I set the option in the model
             because I will udpate the model and add other feature selection strategy sucn as ID3, 
            CART( this one support regression) .

Random Foreset is a classical algorithm of supervised learning. It is consisted by
a big number of decision trees, which is also a classical algorithm of supervised learning
algorithm.
When we input the dataset into the model, it will choose different combination of features and asign
them to one tree, each one tree will usding these features to find out the special relation of these 
features and the label to accuqire the ability to make prediction. 
When we input a data whose label is unknown, the data's features will be splited up in the same way as
 we train the trees, the trees will give their prediction based on the nodes it got while training, 
then the forest will return a summary based on the results it gets from all the trees. Which means 
the trees inside the forest will give tons of answers but what we get is the one which has the most votes.
```


Overfitting and Solution:
```
Random Forest also has the overfitting problem. If you want, you can get answer from all the features 
even though some of them are not important and maybe some times a noise in the data set. Its easy to have
a one hundred percision on training data and the result might be some leaf nodes only have one data point. 
Given that we need to train so many trees and we want our model have a good performance on test data, 
thus we need to restrict the growing of the trees to have a better generalization ability.

For random forest, we usually use two ways to achieve this target. One is pre-pruning (预剪枝), which means 
we will set the depth of each tree so that we could avoid doing too much classification. The another one is
post-pruning(后剪枝), which means we reduce the depth and the number of nodes to avoid overfitting. In this model,
 we use pre-pruning to do this.
```

### Decision Tree
```
The function and theory of Decision Tree is quite simple. For instance, to predict human's power,
we choose sex as the feature. Then the model we get might be the number of big power number concentrated 
in node male. There definitely be less powerful data in male and powerful data in female, but our result 
is based on voting, the minority opinion would be neglected. Thus we cannot have a one hundred accuracy 
because we assume all male are powerful than female. On the other hand, the accuracy would not be low, 
so we could say it is not a perfect model but we could use it as a reference. If we want to impove it,
 we can keep category after sex, like using age, health condition, profession, etc. 
```

### Bagging Algorithm:

```
Given a dataset which has n data, we will select n' data randomly from it and repeat doing this for
m times, so that we will have m models, each model will gives its own prediction, we can get the final
result by calculate the mean value(regression) or voting(classification).
```

Typically, for a k feature dataset, if we are doing classification, we can use k^(1/2) features for each 
partition. If we are doing regression, then k/3 is recommended, at least more than 5. 

### C4.5:
```
    while this node is not pure:
        1. calculate its category entropy info(D) (based on category——label) =  sigma( -num/all * log2(num/sum)) 
            e.g. ten people, 4 people sleep before 10 and 6 don't, there are 3 old man put of 4, 1 out of 6
                then info(sleep) will be - 4/10 * log2(4/10) - 6/10 * log2(6/10) 
        2. calculate its attribution entropy info(Ai) (based on the category value under attribute) 
            = sigma(-num(label)/all * ( sigma( -type1/num(lable) * log2(type1/num(lable))))
            e.g. info(old/young) = - 4/10 * (-3/4 * log2(3/4) - 1/4 *log2(1/4)) - 6/10 * (-1/6 * log2(1/6)) - 5/6 * log2(5/6)) 
        3. calculate gain for each attribution Gain(Ai) = Info(D) - Info(Ai) = 1 - 2
        4. calculate the categorical information measurement of each attribution 
            H(Ai) = sigma(- num(a)/ all * log2(num/all))
            e.g. H(old/young) = - 4 / 10 * log2(4 / 10) - 6 / 10 * log2(6/10)
        5. calculate the info-gain-ratio of each attribution IGR = Gain(Ai) / H(Ai)

    after the process, we will choose the attribute which has the highest IGR, and if the node after spliting 
    is pure(no other label). It will be a leaf node.
         ```
