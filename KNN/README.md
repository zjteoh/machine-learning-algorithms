# How to use KNN

1. This is an easy to use k-nearest-neighbors implementation created from scratch.
2. It has two hyperparameters namely distance functions and k value
3. Normalization is provided if needed
4. You can choose between three types of distance functions: 'euclidean', 'gaussian_kernel' and 'inner_product'
5. Split is the ratio between training and validation/testing. Default split = 0.85
6. max-k is the largest k nearest neighbors to be considered. Default max_k = 10
7. Features should be in List[List[int]] form, where every row represents a data point.
8. Labels should be in List[any] form, where each element represents the classification.

# To initialize k-NN with maximum k = 20 and split-ratio of 0.8
```
knn = k_nearest_neighbors(max_k=20, split=0.8)
```

## To read from specified filename:
```
knn.train('iris_set.csv')
```

## To read data as parameters:
```
knn.train(x=features, y=labels)
```

## To change the default distance function (euclidean):
```
knn.train(..., distance_function='gaussian_kernel')
```

## To normalize the data (L2-Norm)
```
knn.train(...,normal=True)
```

## To predict new features
```
predicted_labels = knn.predict(features_to_predict)
```
