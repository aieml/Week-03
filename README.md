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







Suppose P1 is the point, for which label needs to predict. 
First, you find the k closest point to P1 and then classify points by majority vote of its k neighbors. 
Each object votes for their class and the class with the most votes is taken as the prediction. 
For finding closest similar points, you find the distance between points using distance measures such as Euclidean distance, Hamming distance, Manhattan distance and Minkowski distance.

