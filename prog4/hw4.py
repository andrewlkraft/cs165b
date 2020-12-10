import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numbers


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
            p = np.random.randint(low=0,high=N,size=N)
            clf = DecisionTreeClassifier(random_state=0, max_depth=self.max_depth)
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
        weights = np.ones(N)
        for i in range(self.n_classifiers):
            # create decision stump
            clf = DecisionTreeClassifier(random_state=0, max_depth=self.max_depth, splitter='random')

            clf.fit(X, y, sample_weight=weights)
            # calculate combined weights of incorrectly labeled samples
            incorrect = 0
            predicted = clf.predict(X)
            for j in range(N):
                if predicted[j] != y[j]:
                    incorrect += weights[j]
            
            weighting_constant = np.sqrt((N - incorrect) / incorrect)

            self.alpha.append(np.log(weighting_constant))
            self.ensemble.append(clf)

            # update weight vector
            Z = 0
            for j in range(N):
                if predicted[j] != y[j]:
                    weights[j] *= weighting_constant
                else:
                    weights[j] /= weighting_constant
                Z += weights[j]

            # normalize weight vector so its sum is N
            for j in range(N):
                weights[j] *= N / Z
    
    def test(self, X):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        N, D = np.shape(X)
        votes_by_class = np.empty((N,2))
        for i in range(self.n_classifiers):
            prediction_vector = self.ensemble[i].predict(X)
            votes_by_class[:,1] += self.alpha[i] * prediction_vector
            votes_by_class[:,0] += self.alpha[i] * (np.ones(N) - prediction_vector)

        prediction = np.empty(N)
        for n in range(N):
            if votes_by_class[n,1] > votes_by_class[n,0]:
                prediction[n] = 1
            else:
                prediction[n] = 0

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
