import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    '''
    true_positive = false_positive = false_negative = 0

    for rl, pl in zip(real_labels, predicted_labels):
        if rl == pl == 1:
            true_positive += 1
        elif rl == 0 and pl == 1:
            false_positive += 1
        elif rl == 1 and pl == 0:
            false_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)

    return f1
    '''

    return 2 * sum(np.multiply(real_labels, predicted_labels)) / (sum(real_labels) + sum(predicted_labels))


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        diff = np.subtract(point1, point2)
        return np.cbrt(sum(np.power(np.absolute(diff), 3)))


    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        diff = np.subtract(point1, point2)
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.dot(point1, point2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        a = np.dot(point1, point2)
        b = np.sqrt(np.dot(point1, point1)) * np.sqrt(np.dot(point2, point2))
        return 1 - (a / b)

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        diff = np.subtract(point1, point2)
        diff = np.dot(diff, diff)
        diff = -0.5 * diff
        return -np.exp(diff)


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None      

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        def dist_func_cmp(elem):
            alphabet_list = ['e','m','g','i','c']
            numbers = []
            numbers.append(alphabet_list.index(elem[1]))
            return numbers

        results = []

        for dist_func in distance_funcs:
            for k in range(1,31,2):
                model = KNN(k, distance_funcs[dist_func])
                model.train(x_train, y_train)
                y_pred = model.predict(x_val)
                accuracy = f1_score(y_val, y_pred)
                results.append((accuracy, dist_func[0], k, model, dist_func))

        # prioritize accuracy
        results.sort(key=lambda res : res[0], reverse=True)
        for i, res in enumerate(results):
            if res[0] != results[0][0]:
                results = results[:i]
                break

        # prioritize distance function
        results.sort(key=dist_func_cmp)
        for i, res in enumerate(results):
            if res[1] != results[0][1]:
                results = results[:i]
                break


        # prioritize k
        results.sort(key=lambda res : res[2])

        best = results[0]
        self.best_k = best[2]
        self.best_model = best[3]
        self.best_distance_function = best[4]

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        def dist_func_cmp(elem):
            alphabet_list = ['e','m','g','i','c']
            numbers = []
            numbers.append(alphabet_list.index(elem[1]))
            return numbers

        results = []

        for dist_func in distance_funcs:
            for scale in scaling_classes:
                scaler = scaling_classes[scale]()
                scaled_x_train = scaler(x_train)
                scaled_x_val = scaler(x_val)
                for k in range(1,31,2):
                    model = KNN(k, distance_funcs[dist_func])
                    model.train(scaled_x_train, y_train)
                    y_pred = model.predict(scaled_x_val)
                    accuracy = f1_score(y_val, y_pred)
                    results.append((accuracy, dist_func[0], k, model, dist_func, scale))

        # prioritize by accuracy
        results.sort(key=lambda res : res[0], reverse=True)
        for i, res in enumerate(results):
            if res[0] != results[0][0]:
                results = results[:i]
                break

        # prioritize by normalization
        results.sort(key=lambda res : res[5])
        for i, res in enumerate(results):
            if res[5] != results[0][5]:
                results = results[:i]
                break

        # prioritize distance function
        results.sort(key=dist_func_cmp)
        for i, res in enumerate(results):
            if res[1] != results[0][1]:
                results = results[:i]
                break

        # prioritize k
        results.sort(key=lambda res : res[2])

        best = results[0]
        self.best_k = best[2]
        self.best_model = best[3]
        self.best_distance_function = best[4]
        self.best_scaler = best[5]


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        return [feature/np.sqrt(np.dot(feature, feature)) if np.any(feature) else feature for feature in features]


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.called = False
        self.min = []
        self.max = []

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        res = []

        if not self.called:
            for i in range(len(features[0])):
                curr_min = float('inf')
                curr_max = float('-inf')
                for feature in features:
                    curr_min = min(curr_min, feature[i])
                    curr_max = max(curr_max, feature[i])
                self.min.append(curr_min)
                self.max.append(curr_max)

            self.called = True

        for feature in features:
            temp = []
            for feat, _min, _max in zip(feature, self.min, self.max):
                scaled = None
                if _max - _min != 0:
                    scaled = (feat - _min) / (_max - _min)
                else:
                    scaled = 0
                temp.append(scaled)
            res.append(temp)

        return res








        
