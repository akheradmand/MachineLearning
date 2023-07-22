Assignment 42
#KNN
#ANSUR II dataset

1- Download dataset from kaggle or openlab website.

2- Preprocess dateset for converting unit of weight, unit of height and datatype of gender.
3- Show heights for women and men on same plot.
    A. Why is the data of men higher than the data of women?
    ‌B. Why is the data of men more right than the data of women?

4- Split dataset to train and test datasets (%80 for train and %20 for test):

    from sklearn.model_selection import train_test_split
5- Implement and fit your object oriented KNN algorithm on the train dataset.
6- Evaluate your KNN algorithm on the test dataset with different values of k = 3, 5, 7, ...
and write accuracy results as a table in readme.md.

7- Calculate confusion matrix for test dataset.

8- Fit the scikit-learn KNN algorithm on the train dataset:

from sklearn.neighbors import KNeighborsClassifier
9- Evaluate the scikit-learn KNN algorithm on the test dataset. Make sure your accuracy is equal to scikit-learn's accuracy.

10- Calculate confusion matrix using scikit-learn:

**************************
k=3
accuracy=0.8138385502471169
k=5
accuracy=0.8154859967051071
k=7
accuracy=0.8294892915980231
k=9
accuracy=0.8327841845140033

As we can see, the estimation accuracy increases with the increase of "k"
**************************
evaluate accuracy using scikit-learn KNN algorithm:
n_neighbors=3
accuracy=0.8121911037891268
n_neighbors=5
accuracy=0.8171334431630972
n_neighbors=7
accuracy=0.8303130148270181
n_neighbors=9
accuracy=0.8327841845140033
