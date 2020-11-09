import numpy as np
from collections import Counter

# Below is a class that implement the binary tree search. 
# It shows how recursion is done in a typical binary tree structure. 
# You may apply similar ideas (recursion) in CART as you need to recursively split the left and right node until a stop criterion is met.
# For your reference only. 
NULL = 0 
class bsearch(object):
    '''
    binary search tree, with public functions of search, insert and traversal
    '''
    def __init__ (self, value) :
        self.value = value
        self.left = self.right = NULL

    def search(self, value) :
        if self.value==value :
            return True 
        elif self.value>value :
            if self.left==NULL :
                return False 
            else:
                return self.left.search(value)
        else :		
            if self.right==NULL : 
                return False 
            else :
                return self.right.search(value)

    def insert(self, value) :
        if self.value==value :
            return False 
        elif self.value>value :
            if self.left==NULL :
                self.left = bsearch(value)
                return True 
            else :
                return self.left.insert(value)
        else :
            if self.right==NULL :
                self.right = bsearch(value)
                return True 
            else :
                return self.right.insert(value)

    def inorder(self)  :
        if self.left !=NULL  :
            self.left.inorder()
        if self != NULL : 
            print (self.value, " ", end="")
        if self.right != NULL : 
            self.right.inorder()


# -------------------------------Main code starts here-------------------------------------#
class TreeNode(object):
    '''
    A class for storing necessary information at each tree node.
    Every node should be initialized as an object of this class. 
    '''
    def __init__(self, d=None, threshold=None, l_node=None, r_node=None, label=None, is_leaf=False, gini=None, n_samples=None):
        '''
        Input:
            d: index (zero-based) of the attribute selected for splitting use. int
            threshold: the threshold for attribute d. If the attribute d of a sample is <= threshold, the sample goes to left 
                       branch; o/w right branch. float
            l_node: left children node/branch of current node. TreeNode
            r_node: right children node/branch of current node. TreeNode
            label: the most common label at current node. int/float
            is_leaf: True if this node is a leaf node; o/w False. bool
            gini: stores gini impurity at current node. float
            n_samples: number of samples at current node. int
        '''
        self.d = d
        self.threshold = threshold
        self.l_node = l_node
        self.r_node = r_node
        self.label = label
        self.is_leaf = is_leaf
        self.gini = gini
        self.n_samples = n_samples


def load_data(fdir):
    '''
    Load attribute values and labels from a npy file. 
    Data is assumed to be stored of shape (N, D) where the first D-1 cols are attributes and the last col stores the labels.
    Input:
        fdir: file directory. str
    Output:
        data_x: feature vector. np ndarray
        data_y: label vector. np ndarray
    '''
    data = np.load(fdir)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class CART(object):
    '''
    Classification and Regression Tree (CART). 
    '''
    def __init__(self, max_depth=None):
        '''
        Input:
            max_depth: maximum depth allowed for the tree. int/None.
        Instance Variables:
            self.max_depth: stores the input max_depth. int/inf
            self.tree: stores the root of the tree. TreeNode object
        '''
        self.max_depth = float('inf') if max_depth is None else max_depth 
        self.tree = None
        ###############################
        # TODO: your implementation
        # Add anything you need
        ###############################
        pass

    def train(self, X, y):
        '''
        Build the tree from root to all leaves. The implementation follows the pseudocode of CART algorithm.  
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        ###############################
        # TODO: your implementation
        ###############################
        pass

    def test(self, X_test):
        '''
        Predict labels of a batch of testing samples. 
        Input:
            X_test: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        pass


    def visualize_tree(self):
        '''
        A simple function for tree visualization. 
        Note that this function assumes a class variable called self.tree that stores the root node.
        If your implementation does not satisfy this pre-requisite, this function may generate false visualization.
        You can modify this function to make it fit your implementation.
        
        In your final submission to gradescope, you should avoid calling this function!
        '''
        print('ROOT: ')
        def print_tree(tree, indent='\t|', dict_tree={}, direct='L'):
            if tree.is_leaf == True:
                dict_tree = {direct: str(tree.label)}
            else:
                print(indent + 'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold))

                if tree.l_node.is_leaf == True:
                    print(indent + 'L -> label: %d' % tree.l_node.label)
                else:
                    print(indent + "L -> ",)
                a = print_tree(tree.l_node, indent=indent + "\t|", direct='L')
                aa = a.copy()

                if tree.r_node.is_leaf == True:
                    print(indent + 'R -> label: %d' % tree.r_node.label)
                else:
                    print(indent + "R -> ",)
                b = print_tree(tree.r_node, indent=indent + "\t|", direct='R')
                bb = b.copy()

                aa.update(bb)
                stri = indent + 'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold)
                if indent != '\t|':
                    dict_tree = {direct: {stri: aa}}
                else:
                    dict_tree = {stri: aa}
            return dict_tree
        try:
            if self.tree is None:
                raise RuntimeError('No tree has been trained!')
        except:
            raise RuntimeError('No self.tree variable!')
        _ = print_tree(self.tree)

        
def GridSearchCV(X, y, depth=[1, 40]):
    '''
    Grid search and cross validation.
    Try different values of max_depth to observe the performance change. 
    Apply 5-fold cross validation to find the best depth. 
    Input:
        X: full training dataset. Not split yet. np ndarray
        y: full training labels. Not split yet. np ndarray
        depth: [minimum depth to consider, maximum depth to consider]. list of integers
    Output:
        best_depth: the best max_depth value from grid search results. int
        best_acc: the validation accuracy corresponding to the best_depth. float
        best_tree: a decision tree object that is trained with 
                   full training dataset and best max_depth from grid search. instance
    '''
    ###############################
    # TODO: your implementation
    ###############################
    return best_depth, best_acc, best_tree

# main
# NOTE: Do not change anything below
X_train, y_train = load_data('winequality-red-train.npy')
best_depth, best_acc, best_tree = GridSearchCV(X_train, y_train, [1, 40])
print('Best depth from %5-fold cross validation: %d' % best_depth)
print('Best validation accuracy: %.5f' % (best_acc))
