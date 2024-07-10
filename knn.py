import pandas as pd 
import numpy as np 


prediction = []

def distanceFunction(distanceMethod, instance1, instance2):

    with np.errstate(invalid='ignore'):
        if distanceMethod == "euclidean":
            return np.linalg.norm(instance1 - instance2)
        elif distanceMethod == "manhattan":
            return np.sum(np.abs(instance1 - instance2))
        elif distanceMethod == "chebyshev":
            return np.max(np.absolute(instance1 - instance2))

def fill_missing_features(existing_data, test_data, k, distance_method, distance_threshold, weighted_voting):

    test_data = test_data.to_numpy()
    existing_data = existing_data.to_numpy()

    indexInTest_Data = 0
    for i in test_data:

        testInstance = i
        distanceMatrix = np.array(0)
        missingIndex = np.where(np.isnan(testInstance))[0][0]
        testInstance[missingIndex] = 0 # nan becomes 0 now

        for j in existing_data:

            distance = distanceFunction(distance_method, testInstance, j)
            if distance_threshold == None:
                distanceMatrix = np.append(distanceMatrix, distance)

            else:
                if distance <= distance_threshold:
                    distanceMatrix = np.append(distanceMatrix, distance)
                else:
                    distanceMatrix = np.append(distanceMatrix, float('inf'))
                    pass

        distanceMatrix = np.delete(distanceMatrix, 0)
        firstKLabel = np.argpartition(distanceMatrix, k)

        values = []
        for j in range(k):
            values.append(existing_data[firstKLabel[j]][missingIndex])

        predictedValue = 0
        if weighted_voting == False:
            predictedValue = sum(values)/len(values)

        else: # weighted
            totalSum = 0
            with np.errstate(divide='ignore', invalid='ignore'):
                for j in range(k):
                    totalSum += values[j] / (1/(distanceMatrix[firstKLabel[j]]* distanceMatrix[firstKLabel[j]]))

            predictedValue = totalSum/k

        test_data[indexInTest_Data][missingIndex] = predictedValue

        indexInTest_Data += 1

    columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30']
    dataFrame = pd.DataFrame(test_data, columns=columns)

    return dataFrame  

def knn(existing_data, test_data, k, distance_method, re_training, distance_threshold, weighted_voting):
    
    test_data = test_data.to_numpy()
    existing_data = existing_data.to_numpy()
    
    global prediction
    if re_training == True:
        for i in range(len(test_data)):
            test_data[i][0] = prediction[i]
           
        existing_data = np.concatenate([existing_data, test_data])

    predictions = []

    for i in test_data:

        testInstance = i    
        labels = []
        distanceMatrix = np.array(0)

        for j in existing_data:

            distance = distanceFunction(distance_method, testInstance, j)
            if distance_threshold == None:
                distanceMatrix = np.append(distanceMatrix, distance)

            else:
                if distance <= distance_threshold:
                    distanceMatrix = np.append(distanceMatrix, distance)
                else:
                    distanceMatrix = np.append(distanceMatrix, float('inf'))
                    pass

        distanceMatrix = np.delete(distanceMatrix, 0)
        firstKLabel = np.argpartition(distanceMatrix, k)

        for j in range(k):
            labels.append(existing_data[firstKLabel[j]][0])

        scoreOne = 0
        scoreZero = 0

        if weighted_voting == False:

            if labels.count(1.0) > labels.count(0.0): ## old version was labels[:k]
                predictions.append(1)
            else:
                predictions.append(0)

        else: # weighted
            for i in range(k):
                if existing_data[firstKLabel[j]][0] == 1.0:
                    
                    with np.errstate(divide='ignore'):
                        scoreOne += 1/(distanceMatrix[firstKLabel[j]]* distanceMatrix[firstKLabel[j]])
                else:
                    with np.errstate(divide='ignore'):
                        scoreZero += 1/(distanceMatrix[firstKLabel[j]]* distanceMatrix[firstKLabel[j]])
            
            if scoreOne > scoreZero:
                predictions.append(1)
            else:
                predictions.append(0)

    return predictions


#test_data = pd.read_csv('data/test.csv')
existing_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test_with_missing.csv')

# do not remove this if you want to do retraining at first
prediction = list(test_data['label'])


for k in [3, 5, 10]:
    for distance_method in ['euclidean', 'manhattan', 'chebyshev']:
        for re_training in [False, True]:
            for distance_threshold in [None, 50]:
                for weighted_voting in [True, False]:
                    filled_data = fill_missing_features(existing_data, test_data, k, distance_method, distance_threshold, weighted_voting)
                    predictions = knn(existing_data, filled_data, k, distance_method, re_training, distance_threshold, weighted_voting)
                    accuracy = np.sum(predictions == filled_data.iloc[:, 0]) / len(filled_data)
                    print("k=", k, ", distance_method=", distance_method, ", re_training=", re_training, ", distance_threshold=", distance_threshold, ", weighted_voting=", weighted_voting)
                    print("Accuracy: ", accuracy)