import numpy as np
import utils as Util


class ID3():
    def __init__(self):
        self.rootNode = None

    def train(self, features, classes):
        # features: List[List[float]], classes: List[int]
        # init
        assert (len(features) > 0)
        numClasses = np.unique(classes).size

        # build the tree
        self.rootNode = Node(features, classes, numClasses)
        if self.rootNode.canSplit:
            self.rootNode.splitNode()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        class_pred = []
        for feature in features:
            pred = self.rootNode.predict(feature)
            class_pred.append(pred)
        return class_pred


class Node(object):
    def __init__(self, features, classes, numClasses):
        # features: List[List[any]], classes: List[int], numClasses: int
        self.features = features
        self.classes = classes
        self.children = []
        self.numClasses = numClasses

        # find the most common class in current node
        count_max = 0
        for label in np.unique(classes):
            if self.classes.count(label) > count_max:
                count_max = classes.count(label)
                self.classMajority = label
        
        # splitable is false when all features belongs to one class
        # or when there are no more features to be split
        
        if np.array(features).size == 0:
            self.canSplit = False
        else:
            if len(np.unique(classes)) < 2:
                self.canSplit = False
            else:
                self.canSplit = True            

        self.dimSplit = None  # the index of the feature to be split

        self.featureUniqSplit = None  # the possible unique values of the feature to be split

    def splitNode(self):
        value,counts = np.unique(self.classes, return_counts=True)
        norm_counts = counts / counts.sum()
        currentEnt = sum(-norm_counts * np.log2(norm_counts))

        # label dict
        lab_dict = dict()
        lab_idx = 0
        for lab in self.classes:
            if not lab in lab_dict:
                lab_dict[lab] = lab_idx
                lab_idx += 1

        arrT = np.array(self.features).T.tolist()

        info_gain_res = []

        for i, feat in enumerate(arrT):
            # feat dict
            feat_dict = dict()
            feat_idx = 0
            for f in feat:
                if not f in feat_dict:
                    feat_dict[f] = feat_idx
                    feat_idx += 1

            branches = [([0] * self.numClasses) for i in range(len(feat_dict))]
            
            for f,l in zip(feat, self.classes):
                branches[feat_dict[f]][lab_dict[l]] += 1

            info_gain = Util.informationGain(currentEnt, branches)
            info_gain_res.append([info_gain, len(branches), -i])

        info_gain_res = sorted(info_gain_res, key=lambda x : (x[0], x[1], x[2]), reverse=True)

        self.dimSplit = -info_gain_res[0][2]
        self.featureUniqSplit = sorted(np.unique(arrT[self.dimSplit]))

        for uniq in self.featureUniqSplit:
            new_feat = []
            new_lab = []
            for i, f in enumerate(arrT[self.dimSplit]):
                if f == uniq:
                    new_feat.append(self.features[i][:self.dimSplit] + self.features[i][self.dimSplit+1:])
                    new_lab.append(self.classes[i])

            new_node = Node(new_feat, new_lab, np.unique(new_lab).size)
            self.children.append(new_node)

        for child in self.children:
            if child.canSplit:
                child.splitNode()


    def predict(self, feature):
        # feature: List[any]
        # return: int
        currNode = self

        while len(currNode.children) > 0:
            feat = feature[currNode.dimSplit]
            feature = feature[:currNode.dimSplit] + feature[currNode.dimSplit+1:]
            found_path = False
            for i, f in enumerate(currNode.featureUniqSplit):
                if f == feat:
                    currNode = currNode.children[i]
                    found_path = True
                    break
            if not found_path:
                return currNode.classMajority

        return currNode.classMajority


