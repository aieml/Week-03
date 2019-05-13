from sklearn import datasets    #importing datasets

iris=datasets.load_iris()       #loading iris flower dataset into iris

data=iris.data      #iris flower data, 150x4 array will be loaded to data
target=iris.target  #iris flower targets, 150x1 array will be loaded to target

#print(data)
#print(target)
#features=iris.feature_names
#print(features)

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.5)
#print(test_target)

from sklearn.neighbors import KNeighborsClassifier #load KNN classifer

clsfr=KNeighborsClassifier(n_neighbors=3)    #KNN classifier is loaded to clsfr

clsfr.fit(train_data,train_target)  #training the ML algorithm(KNN)

results=clsfr.predict(test_data)

print('Predicted:',results)
print('Actual:',test_target)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(test_target,results)

print('accuracy:',accuracy)


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #Axes3D used for 3 axes graphs

fig=plt.figure()      #initialize the 3d Graph
ax=fig.add_subplot(111,projection='3d')     #adding 3 axes to the fig graph, 111-xyz true


for i in range(0,len(train_target)):

    if(train_target[i]==0):     #if target is setosa

        ax.scatter(train_data[i][0],train_data[i][1],train_data[i][2],c='g')
        
    elif(train_target[i]==1):   #if target is verginica

        ax.scatter(train_data[i][0],train_data[i][1],train_data[i][2],c='r')

    elif(train_target[i]==2):   #if target is versicolor

        ax.scatter(train_data[i][0],train_data[i][1],train_data[i][2],c='b')

print(results[0],test_target[0])

ax.scatter(test_data[0][0],test_data[0][1],test_data[0][2],c='c',marker='x')

plt.show()
