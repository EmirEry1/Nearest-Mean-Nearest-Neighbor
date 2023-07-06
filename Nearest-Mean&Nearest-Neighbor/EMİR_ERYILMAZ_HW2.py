import numpy as np
import math

def findMeans(train_set):
    means = [ [ 0 for i in range(4) ] for j in range(3) ]
    for i in range(len(train_set[0])-1):
        sums = [0.0,0.0,0.0]
        counts = [0.0,0.0,0.0]
        for row in train_set:
            sums[int(row[len(row)-1])] += row[i]
            counts[int(row[len(row)-1])] +=1 
        for y in range(3):
            means[y][i] = sums[y]/(counts[y])
    return means

def findDistance(x,y):
    distance_squared = 0.0
    for dim in range(len(x)):
        distance_squared +=(x[dim]-y[dim])**2
    return math.sqrt(distance_squared)

def findNearestMean(test_set, means):
    classifications = [0 for i in range(len(test_set))]
    for point in range (len(test_set)):
        min_distance = math.inf
        min_class = None
        for i in range(len(means)):
            dist = findDistance(test_set[point],means[i])
            if min_distance > dist :
                min_class = i
                min_distance = dist
        classifications[point] = min_class
    return classifications

def findNearest(train_set, test_set): 
    train_set_modified = train_set
    classifications = [0 for i in range(len(test_set))]
    for point1 in range(len(test_set)):
        min_distance = math.inf
        closest_point_class = None
        for point2 in range(len(train_set_modified)):
            dist = findDistance(train_set_modified[point2,: len(train_set_modified[0])-1], test_set[point1, : len(test_set[0])-1])
            if dist < min_distance:
                min_distance = dist
                closest_point_class = train_set_modified[point2,len(train_set_modified[0])-1]
        classifications[point1] = closest_point_class
    return classifications

def calculatePredictionRatio(confusion_matrix):
    number_of_correct_pred = 0
    number_of_incorrect_pred = 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            if i == j:
                number_of_correct_pred+=confusion_matrix[i][j]
            else:
                number_of_incorrect_pred +=confusion_matrix[i][j]
    return float(number_of_incorrect_pred)/(number_of_correct_pred+number_of_incorrect_pred)

train_set = np.loadtxt(fname="train_iris.csv", delimiter=',', dtype=np.float32, skiprows=1) # I have deleted the first row of the 
test_set = np.loadtxt(fname="test_iris.csv", delimiter=',', dtype=np.float32, skiprows=1)

means = findMeans(train_set)
classifications = findNearestMean(test_set[:,:len(test_set[0])-1],means)
confusion_matrix_nearest_mean = [ [ 0 for prediction in range(3) ] for actual in range(3) ]

for point in range(len(test_set)):
    confusion_matrix_nearest_mean[classifications[point]][int(test_set[point,len(test_set[0])-1])] +=1

classifications_neighbor = findNearest(train_set,test_set)

confusion_matrix_nearest_neighbor = [ [ 0 for prediction in range(3) ] for actual in range(3) ]
for point in range(len(test_set)):
    confusion_matrix_nearest_neighbor[int(classifications_neighbor[point])][int(test_set[point,len(test_set[0])-1])] +=1

print("Here is the means matrix:")
print(means)

print("Here is the confusion matrix for the nearest mean algorithm:")
print(confusion_matrix_nearest_mean)
print("Here is the confusion matrix for the nearest neighbor algorithm:")
print(confusion_matrix_nearest_neighbor)

print("Here is the error ratio for the nearest mean algorithm:")
print(calculatePredictionRatio(confusion_matrix_nearest_mean))
print("Here is the error ratio for the nearest neighbor algorithm:")
print(calculatePredictionRatio(confusion_matrix_nearest_neighbor))