# How to use KNN

1. This is an easy to use k-nearest-neighbors implementation created from scratch.
2. It has two hyperparameters namely distance functions and k value
3. Normalization is provided if needed

## EXAMPLE:
```
knn = k_nearest_neighbors(max_k = 20, split=0.7)
knn.train(features, labels)
predicted_y = knn.predict(feature_to_predict)
```
> max_k = Maximum k neighbors to be considered (default k = 10)

> split = Split ratio between validation and testing. (default split = 0.8) 
        **Split must be between 0.5 ~ 0.8**
