# Week-03
## Classification Machine Learning Algorithms Part I - K Nearest Neighbors Classifier

In Week 03 we discussed about the KNN Classifier, which is one of the most simplest Supervised Type Classification Machine Learning Algorithm

### KNN Classifier

- K Nearest Neighbor(KNN) is a very simple, easy to understand, versatile and one of the topmost machine learning algorithms. 
- KNN used in the variety of applications such as finance, healthcare, political science, handwriting detection, image recognition and video recognition. 
- In Credit ratings, financial institutes will predict the credit rating of customers
- In loan disbursement, banking institutes will predict whether the loan is safe or risky. 
- In political science, classifying potential voters in two classes will vote or won’t vote. 
KNN algorithm used for both classification and regression problems. KNN algorithm based on feature similarity approach.

### Functionality

- In KNN, K is the number of nearest neighbors. The number of neighbors is the core deciding factor. 
- K is generally an odd number if the number of classes is 2. When K=1, then the algorithm is known as the nearest neighbor algorithm. This is the simplest case. 
- Suppose P1 is the point, for which label needs to predict. First, you find the one closest point to P1 and then the label of the nearest point assigned to P1.

<img src="Images/Picture1.png" width="400">

- Suppose P1 is the point, for which label needs to predict. 
- First, you find the k closest point to P1 and then classify points by majority vote of its k neighbors. 
- Each object votes for their class and the class with the most votes is taken as the prediction. 
- For finding closest similar points, you find the distance between points using distance measures such as Euclidean distance, Hamming distance, Manhattan distance and Minkowski distance.

<img src="Images/Picture5.png" width="400">

<img src="Images/Picture4.png" width="400">

#### How to find the distance Between Points - Euclidean Distance

<img src="Images/Picture2.png" width="400">
<img src="Images/Picture3.jpg" width="400">

#### K value

- Research has shown that no optimal number of neighbors suits all kind of data sets. 
- Each dataset has it's own requirements. 
- In the case of a small number of neighbors, the noise will have a higher influence on the result, and a large number of neighbors make it computationally expensive.
- Research has also shown that a small amount of neighbors are most flexible fit which will have low bias but high variance and a large number of neighbors will have a smoother decision boundary which means lower variance but higher bias.
- K value can be set in scikit learn as mentioned below

```python
from sklearn.neighbors import KNeighborsClassifier
clsfr=KNeighborsClassifier(n_neighbors=3)
```

- Generally, Data scientists choose as an odd number if the number of classes is even. 
- You can also check by generating the model on different values of k and check their performance. You can also try Elbow method

<img src="Images/Picture6.png" width="400">

#### Points to Remember

- The training phase of K-nearest neighbor classification is much faster compared to other classification algorithms.  
- KNN can be useful in case of nonlinear data. It can be used with the regression problem. 
- Output value for the object is computed by the average of k closest neighbors value. 
- The testing phase of K-nearest neighbor classification is slower and costlier in terms of time and memory. 
- Euclidean distance is sensitive to magnitudes, therefore KNN also not suitable for large dimensional data.  
