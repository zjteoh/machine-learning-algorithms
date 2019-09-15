# How to use KNN

1. This is an easy to use k-nearest-neighbors implementation created from scratch.
2. It has two hyperparameters namely distance functions and k value
3. Normalization is provided if needed

## To read from specified filename:
```
knn = k_nearest_neighbors()
knn.train('iris_set.csv')
predicted_y_list = knn.predict(features_to_predict) # features_to_predict should be in List[List[int]] format
```

## To read data as parameters:
```
knn = k_nearest_neighbors()
knn.train(x=features, y=labels) # features => List[List[int]], # labels = List[any]
predicted_y_list = knn.predict(features_to_predict) # features_to_predict should be in List[List[int]] format
```

> max_k = Maximum k neighbors to be considered (default k = 10)

> split = Split ratio between training and validation/testing and only considered if filename not given during train()
	 = The testing ratio is always set at 0.05. For example, if split = 0.7, then training ratio is 0.7, validation is
	   0.25 and testing is 0.05
