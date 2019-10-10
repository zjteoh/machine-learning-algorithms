import data
import decisiontree as decision_tree
import utils as Utils


def information_gain_test():
    branch = data.sample_branch_data()
    score = Utils.informationGain(0, branch)
    print('Your information gain: ', score)


def decision_tree_test():
    features, labels = data.sample_decision_tree_data()

    # build the tree
    dTree = decision_tree.ID3()

    dTree.train(features, labels)

    # print
    print('Your decision tree: ')
    Utils.printTree(dTree)

    # data
    X_test, y_test = data.sample_decision_tree_test()

    # testing
    y_est_test = dTree.predict(X_test)
    print('Your estimate test: ', y_est_test)


def pruning_decision_tree_test():
    # load data
    X_train, y_train, X_test, y_test = data.sample_decision_tree_pruning()

    # build the tree
    dTree = decision_tree.ID3()
    dTree.train(X_train, y_train)

    # print
    print('Your decision tree:')
    Utils.printTree(dTree)

    Utils.reducedErrorPruning(dTree, X_test, y_test)
    print('Your decision tree after pruning:')
    Utils.printTree(dTree)

if __name__ == "__main__":
    information_gain_test()
    decision_tree_test()
    pruning_decision_tree_test()
