import numpy as np
from collections import Counter

'''
This is an easy to use k-nearest-neighbors implementation created from scratch.
It has two hyperparameters namely distance functions and k value
Normalization is provided if needed

max_k = maximum k neighbors to be considered (default k = 10)

split = split ratio between validation and testing. (default split = 0.8) 
        ** split must be between 0.5 ~ 0.8 **

EXAMPLE:
knn = k_nearest_neighbors(max_k = 20, split=0.7)
knn.train(features, labels)
predicted_y = knn.predict(feature_to_predict)
'''

class k_nearest_neighbors:
	def __init__(self, max_k = 10, split = 0.8):
		if split > 0.9 or split < 0.5:
			raise ValueError("Split should be between 0.5 to 0.9")

		self.max_k = max_k
		self.split = split
		self.best_k = None
		self.best_distance_function = None
		self.x = None
		self.y = None

	def train(self, x, y):
		# save data
		self.x = x
		self.y = y

		# find best hyperparameters
		x_train = np.array(x)[:int(self.split * len(x)) ,:].tolist()
		x_test = np.array(x)[int(self.split * len(x)): ,:].tolist()

		y_train = y[:int(0.8 * len(y))]
		y_test = y[int(0.8 * len(y)):]

		self.tuning(x_train, y_train, x_test, y_test)

	def predict(self, x, k = None, x_train = None, y_train = None):
		if k == None:
			k = self.best_k

		y_list = self.get_k_nearest_neighbors(x, k, x_train, y_train)
		c = Counter(y_list)
		predicted_y, _ = c.most_common()[0]
		return predicted_y

	def get_k_nearest_neighbors(self, x, k, x_train = None, y_train = None):
		dist_list = []

		for _x, _y in zip(x_train if x_train else self.x, y_train if y_train else self.y):
			dist_list.append([self.get_distance(x, _x), _y])

		dist_list.sort()

		return [y for _, y in dist_list[:k]]


	def get_distance(self, x1, x2):
		def euclidean_distance(x1, x2):
			difference = np.subtract(x1,x2)
			return np.sqrt(np.dot(difference, difference))

		return euclidean_distance(x1, x2)


	def tuning(self, x_train, y_train, x_test, y_test):
		best_acc = 0
		for k in range(1, self.max_k):
			total_correct = 0
			for x, y in zip(x_test, y_test):
				if self.predict(x, k, x_train, y_train) == y:
					total_correct += 1
			accuracy = total_correct / len(y_test)
			if accuracy > best_acc:
				best_acc = accuracy
				self.best_k = k


	def normalization(self):
		raise NotImplementedError


x = [[1,0,1],
     [0,0,0],
     [1,0,1],
     [0,0,0]]

y = [1,0,1,0]

knn = k_nearest_neighbors()
knn.train(x,y)
print(knn.predict([1,0,1]))



