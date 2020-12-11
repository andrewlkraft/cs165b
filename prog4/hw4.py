import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numbers
import matplotlib.pyplot as plt

random = np.random.default_rng()

def load_data(filename):
    data = np.load(filename)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class Bagging(object):
    def __init__(self, n_classifiers, max_depth):
        '''
        Input:
            n_classifiers: number of trees in the ensemble. int
            max_depth: maximum depth allowed for every tree built. It should not exceed 20. int
        '''
        # sklearn.tree.DecisionTreeClassifier()
        self.n_classifiers = n_classifiers
        self.max_depth = max_depth
        self.ensemble = []
    
    def train(self, X, y):
        '''
        Build an ensemble.
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        N, D = np.shape(X)
        for i in range(self.n_classifiers):
            p = random.integers(low=0,high=N,size=N)
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X[p],y[p])
            self.ensemble.append(clf)

   
    def test(self, X):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        N, D = np.shape(X)
        aggregate_prediction = np.zeros((N,self.n_classifiers))
        for i in range(self.n_classifiers):
            aggregate_prediction[:,i] = self.ensemble[i].predict(X)

        prediction, counts = stats.mode(aggregate_prediction, axis=1)
        return prediction.ravel()


class Boosting(object):
    def __init__(self, n_classifiers, max_depth):
        '''
        Input:
            n_classifiers: the maximum number of trees at which the boosting is terminated. int
            max_depth: maximum depth allowed for every tree built. It should not exceed 2. int
        '''
        if max_depth!=1 and max_depth!=2:
            raise ValueError('max_depth can only be 1 or 2!')

        self.n_classifiers = n_classifiers
        self.max_depth = max_depth
        self.ensemble = []
        self.alpha = []

    def train(self, X, y):
        '''
        Train an adaboost.
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        N, D = np.shape(X)
        # initialise weight vector to contain equal values which add up to 1
        weights = np.ones(N) / N

        for i in range(self.n_classifiers):
            # train decision stump with weights
            clf = DecisionTreeClassifier(max_depth=self.max_depth, splitter='random')
            clf.fit(X, y, sample_weight=weights)

            # calculate error rate
            predicted = clf.predict(X)
            err = 0
            for j in range(N):
                if predicted[j] != y[j]:
                    err += weights[j]

            # calculate and store alpha val
            self.alpha.append(1/2*np.log((1 - err) / err))
            self.ensemble.append(clf)

            # reweigh samples based on whether they are misclassified
            Z = 0
            for j in range(N):
                Z += weights[j]
                if predicted[j] != y[j]:
                    weights[j] *= np.exp(self.alpha[i])
                else:
                    weights[j] *= np.exp(-1 * self.alpha[i])

            # normalize weight vector so its sum is N
            for j in range(N):
                weights[j] *= 1 / Z

        plt.plot(self.alpha, 'bo', markersize=2)
        plt.xlabel('classifier #')
        plt.ylabel('alpha')
        plt.title('value of alpha for each epoch')
        plt.show()

        total = sum(self.alpha)
        for i in range(self.n_classifiers):
            self.alpha[i] /= total

    def test(self, X, y=None):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        N, D = np.shape(X)
        prediction = np.zeros(N)

        for i in range(self.n_classifiers):
            prediction += self.alpha[i] * self.ensemble[i].predict(X)

        if y is not None:
            class1, class0 = list(), list()
            for i in range(N):
                if y[i]==0:
                    class0.append(prediction[i])
                else:
                    class1.append(prediction[i])
            

            markersize = 2
            plt.plot(class0, 'bo', label='class 0', markersize=markersize)
            plt.plot(class1, 'ro', label='class 1', markersize=markersize)
            plt.legend()
            plt.xlabel('sample num')
            plt.ylabel('weighted sum')
            plt.show()

        for i in range(N):
            if prediction[i] > .5:
                prediction[i] = 1
            else:
                prediction[i] = 0

        return prediction

# Please do not modify the variable names
X_train, y_train = load_data('winequality-red-train-2class.npy')
X_test, y_test = load_data('winequality-red-test-2class.npy')
bagging = Bagging(50, 20)
boosting = Boosting(1000, 2)
bagging.train(X_train, y_train)
boosting.train(X_train, y_train)
print('Bagging test accuracy: %.5f' % np.mean(bagging.test(X_test)==y_test))
print('AdaBoost test accuracy: %.5f' % np.mean(boosting.test(X_test)==y_test))