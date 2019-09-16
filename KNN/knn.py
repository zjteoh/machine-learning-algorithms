import numpy as np
from collections import Counter
import pandas as pd

'''
This is an easy to use k-nearest-neighbors implementation created from scratch.
It has two hyperparameters namely distance functions and k value
Normalization is provided if needed

max_k = maximum k neighbors to be considered (default k = 10)

split = Split ratio between training and validation/testing (only considered if filename not given during train())
	  = The testing ratio is always set at 0.05. For example, if split = 0.7, then training ratio is 0.7, validation is
	    0.25 and testing is 0.05

For the data, all features should be aligned by column. This means each row represents a single data point

EXAMPLE:
knn = k_nearest_neighbors(max_k = 20, split=0.7)
knn.train('iris_set.csv') # if reading from file
knn.train(x=features, y=labels) # if not reading from file
predicted_y = knn.predict(features_to_predict) # features_to_predict should be in List[List[int]] format
'''

class k_nearest_neighbors:
	'''
	max_k = The largest number of neighbors to be considered. Default set at 10
	split = Split ratio between training and validation/testing
	'''
	def __init__(self, max_k = 10, split = 0.85):
		if split > 0.85 or split < 0.6:
			raise ValueError("Split should be between 0.6 to 0.85")

		self.max_k = max_k
		self.split = split
		self.best_k = None
		self.best_distance_function = None



	'''
	x = List[List[int]] -- feature(s) of data
	  = Each row should represent a data point
	  = Each column represents the same feature for all data points
	y = List(any) -- labels of data

	Initializes the data and finds the best hyperparameters given the set of data
	'''
	def train(self, filename='', x = None, y = None):
		if len(filename) > 0:
			self.filename = filename
			self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.read_and_process_data(filename) 
		elif x != None and y != None:
			arrX = np.array(x)

			self.x_train = arrX[:int(self.split * len(x)) ,:].tolist()
			self.y_train = y[:int(self.split * len(y))]

			self.x_val = arrX[int(self.split * len(x)):int(0.95 * len(x)),:].tolist()
			self.y_val = y[int(self.split * len(y)):int(0.95 * len(y))]

			self.x_test = arrX[int(0.95 * len(x)): ,:].tolist()
			self.y_test = y[int(0.95 * len(y)):]
		else:
			raise ValueError('Either filename must be set OR x and y must be set')

		self.normalize()
		self.tuning()



	'''
	data = List[List[int]] -- feature(s) of data
	     = Each row should represent a data point
	     = Each column represents the same feature for all data points
	
	This function returns the list of predicted labels of the data
	'''
	def predict(self, data):
		return [_predict(x) for x in data]



	'''
	x = feature(s) for a point
	k = the number of nearest neighbors to be considered
	tuning = True during tuning, False otherwise

	Utility function used to predict a single point data. For training (tuning),
	x_val and y_val are used. Else, for x_train and y_train are used.
	'''
	def _predict(self, x, k = None, tuning=False):
		if k == None:
			k = self.best_k

		y_list = self.get_k_nearest_neighbors(x, k, tuning=tuning)

		c = Counter(y_list)
		predicted_y, _ = c.most_common()[0]
		return predicted_y

	

	'''
	x = feature(s) for a point
	k = the number of nearest neighbors to be considered
	tuning = True during tuning, False otherwise

	This function returns the nearest k-neighbors labels. If called during tuning,
	then data used is x_val and y_val. If called during prediction of 
	test or real data, then x_trainand y_train are used instead
	'''
	def get_k_nearest_neighbors(self, x, k, tuning=False):
		dist_list = []

		x_temp = self.x_train if not tuning else self.x_val
		y_temp = self.y_train if not tuning else self.y_val

		for _x, _y in zip(x_temp, y_temp):
			dist_list.append([self.get_distance(x, _x), _y])

		dist_list.sort()

		return [y for _, y in dist_list[:k]]



	'''
	x1 = feature(s) for point x1
	x2 = feature(s) for point x2

	Find the distance between two points. This function has inner functions of various distance functions
	When called by predict() function, it will choose the best distance function based on the hyperparameter
	best_distance_function set by tuning() earlier during training 
	'''
	def get_distance(self, x1, x2):
		def euclidean_distance(x1, x2):
			difference = np.subtract(x1,x2)
			return np.sqrt(np.dot(difference, difference))

		return euclidean_distance(x1, x2)



	'''
	x_train = List[List[int]]
	y_train = List[any]
	x_test = List[List[int]]
	y_test = List[any]

	Function used to tune the best hyperparameters k and distance function
	'''
	def tuning(self):
		best_acc = 0
		for k in range(1, self.max_k):
			total_correct = 0
			for x, y in zip(self.x_test, self.y_test):
				if self._predict(x, k, tuning=True) == y:
					total_correct += 1
			accuracy = total_correct / len(self.y_test)
			if accuracy > best_acc:
				best_acc = accuracy
				self.best_k = k


	

	def normalize(self):
		self.x_train = [x/np.sqrt(np.dot(x,x)) if np.any(x) else x for x in self.x_train]
		self.x_val = [x/np.sqrt(np.dot(x,x)) if np.any(x) else x for x in self.x_val]
		self.x_test = [x/np.sqrt(np.dot(x,x)) if np.any(x) else x for x in self.x_test]





	'''
	Reads data from the given filename

	Returns x_train, y_train, x_val, y_val, x_test and y_test

	Ratio between train, val and test is 0.85, 0.15 and 0.05 respectively
	'''
	def read_and_process_data(self, filename):
	    data = pd.read_csv(filename, low_memory=False, sep=',', na_values='?').values

	    N = data.shape[0]

	    # shuffle data 
	    np.random.shuffle(data)

	    n_train = int(np.round(N * 0.8))
	    n_val = int(np.round(N * 0.15))
	    n_test = N - n_train - n_val

	    # spliting training, validation, and test
	    x_train = np.append([np.ones(n_train)], data[:n_train].T[:-1], axis=0).T
	    y_train = data[:n_train].T[-1].T
	    x_val = np.append([np.ones(n_val)], data[n_train:n_train + n_val].T[:-1], axis=0).T
	    y_val = data[n_train:n_train + n_val].T[-1].T
	    x_test = np.append([np.ones(n_test)], data[-n_test:].T[:-1], axis=0).T
	    y_test = data[-n_test:].T[-1].T
	    return x_train, y_train, x_val, y_val, x_test, y_test 

