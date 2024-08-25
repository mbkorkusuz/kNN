# KNN
## **Introduction**
This repository contains the implementation of well known knn algorithm with missing feature prediction extension.

## **Installation**

1. Clone repository to your local machine
 ````text
git clone https://github.com/mbkorkusuz/kNN.git
````
2. Navigate to the project directory
3. Run `knn.py` script
````text
python3 knn.py 
````
Script will loop through different configurations of knn algorithm, and print the accuracy score for each configuration.

    
## **Example Output**

* k= 3 , distance_method= euclidean , re_training= False , distance_threshold= None , weighted_voting= True

**Accuracy:  0.9764705882352941**

* k= 3 , distance_method= euclidean , re_training= True , distance_threshold= None , weighted_voting= True

**Accuracy:  0.9647058823529412**

* k= 5 , distance_method= euclidean , re_training= False , distance_threshold= 50 , weighted_voting= True

**Accuracy:  0.9411764705882353**

* k= 5 , distance_method= euclidean , re_training= True , distance_threshold= None , weighted_voting= False

**Accuracy:  0.9647058823529412**

* k= 10 , distance_method= euclidean , re_training= False , distance_threshold= None , weighted_voting= True

**Accuracy:  0.9058823529411765**

* k= 10 , distance_method= chebyshev , re_training= True , distance_threshold= 50 , weighted_voting= True

**Accuracy:  0.8823529411764706**
