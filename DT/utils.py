import numpy as np

def informationGain(parentEntropy, featureBranches):
    # parentEntropy: float
    # featureBranches: List[List[int]] number of branches * numClasses
    # return: float
    temp = np.array(featureBranches).T
    weighted_avg = np.sum(temp, axis=0) / np.sum(temp)
    ent = temp / np.sum(temp, axis=0)
    ent = [[-_e * np.log2(_e) if _e > 0 else 0 for _e in e]for e in ent]
    ent = np.array(ent)
    ent = np.sum(np.sum(ent, axis=0) * weighted_avg)
    return parentEntropy - ent


def getAccuracy(decisionTree, x_test, y_test):
    total_correct = 0
    y_pred = decisionTree.predict(x_test)

    for y1, y2 in zip(y_pred, y_test):
        if y1 == y2:
            total_correct += 1

    return total_correct / len(y_test)

def getParentsOfLeafs(root, lis): 
    if len(root.children) == 0:
        return

    for child in root.children:
        getParentsOfLeafs(child, lis)

        # root must be parent of some leaf node
        if len(child.children) == 0 and not root in lis:
            lis.append(root)

def reducedErrorPruning(decisionTree, X_test, y_test):
    # X_test: List[List[any]]
    # y_test: List

    # get all parent of leaf nodes
    while True:
        orig_acc = getAccuracy(decisionTree, X_test, y_test)
        parents = []
        getParentsOfLeafs(decisionTree.rootNode, parents)
        parent_acc_list = []

        for parent in parents:
            parent.canSplit = False
            temp = parent.children
            parent.children = []
            parent_acc_list.append([getAccuracy(decisionTree, X_test, y_test), parent])
            parent.canSplit = True
            parent.children = temp

        parent_acc_list.sort(key=lambda x : x[0], reverse=True)
        elem = parent_acc_list[0]

        if elem[0] >= orig_acc:
            # prune the tree
            elem[1].canSplit = False
            elem[1].children = []

            # nothing else to prune
            if elem[1] == decisionTree.rootNode:
                return
        else:
            return


def printTree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.rootNode
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.classes).tolist()
    for label in label_uniq:
        string += str(node.classes.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.canSplit:
        print(indent + '\tsplit by dim {:d}'.format(node.dimSplit))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.classMajority)
    print(indent + '}')
