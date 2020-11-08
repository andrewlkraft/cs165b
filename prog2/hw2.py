import time, numpy as np, matplotlib.pyplot as plt, sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

CROSSVAL_NUM = 3
BATCH_SIZE = 32

class BinaryClassifier():

    def __init__(self, train_data, train_target):
        """Data is already loaded for you using sklearn.datasets.load_breast_cancer
        utility function in auto-testing part. train_data is your training feature
        data and train_target is your train label data. Hint: you can store both
        training features and target vector as class variables"""
        self.n = train_data.shape[0]
        data = np.c_[ train_data.to_numpy(), np.ones(self.n), train_target.to_numpy() ]
        np.random.shuffle(data)
        self.mean = np.mean(data, axis=0)
        self.stdev = np.std(data, axis=0)
        self.features = data[:,:-1]
        self.labels = data[:,-1]
        for n in range(self.n):
            for k in range(30):
                self.features[n,k] = (self.features[n,k] - self.mean[k]) / self.stdev[k]
        self.w = np.zeros(31)
        self.error = np.inf
        self.l = 0
        self.a = 0
        self.grid = None

    def logistic_training(self, alpha, lam, nepoch, epsilon):
        """Training process of logistic regression will happen here. User will provide
        learning rate alpha, regularization term lam, specific number of training epoches,
        and a variable epsilon to specify pre-mature end condition,
        i.e., if error < epsilon, training stops.
        The specification of these parameters is the same as that in program #1.
        You implementation must include 3-fold validation,
        but you are allowed to hardcode the minibatch size in your program.
        Hint: You can store both weight and bias as class variables,
        so other functions can directly use them"""

        alphas = np.linspace(0.001, 0.01, 10)
        lambdas = np.linspace(1e-9,1e-8, 10)
        rng = np.random.default_rng()

        # grid search for alpha and lambda
        for a in alphas:
            for l in lambdas:

                error = np.inf
                model = np.zeros(31)
                # model 3 times for cross validation
                for i in range(CROSSVAL_NUM):
                    w = 2 * rng.random(31) - np.ones(31)
                    epoch_num = 0
                    w_grad = np.zeros(31)
                    n = 0
                    batch_num = 0
                    while epoch_num < nepoch:
                        # update w_grad if the sample is a member of the training set
                        if n % CROSSVAL_NUM != i:
                            yhat = 1 / (1 + np.exp(-np.dot(self.features[n], w)))
                            w_grad = w_grad - (self.labels[n] - yhat) * self.features[n] + l * w
                            batch_num = batch_num + 1
                        n = n + 1

                        # if we have completed a full batch, add w_grad to w and reset w_grad
                        if batch_num == BATCH_SIZE:
                            batch_num = 0
                            w = w - a * w_grad
                            w_grad = np.zeros(31)
                        
                        # update epoch number if all samples have been used
                        if n == self.n:
                            n = 0
                            epoch_num = epoch_num + 1
                            cerr = 0
                            for cn in range(self.n):
                                if cn % CROSSVAL_NUM == i:
                                    yhat = 1 / (1 + np.exp(-np.dot(self.features[cn], w)))
                                    cerr = cerr - self.labels[cn] * np.log(yhat) - (1 - self.labels[cn]) * np.log(1 - yhat)
                            if cerr < epsilon:
                                break
                    if cerr < error:
                        error = cerr
                        model = w
                
                if error < self.error:
                    self.error = error
                    self.w = w
                    self.l = l
                    self.a = a
        
        # perform one final training session to polish results

        epoch_num = 0
        n = 0
        w_grad = np.zeros(31)
        while epoch_num < nepoch:
            # update w_grad if the sample is a member of the training set
            if n % 10 != 0:
                yhat = 1 / (1 + np.exp(-np.dot(self.features[n], w)))
                w_grad = w_grad - (self.labels[n] - yhat) * self.features[n] + self.l * w
                batch_num = batch_num + 1
            n = n + 1
            # if we have completed a full batch, add w_grad to w and reset w_grad
            if batch_num == BATCH_SIZE:
                batch_num = 0
                w = w - self.a * w_grad
                w_grad = np.zeros(31)
                        
            # update epoch number if all samples have been used
            if n == self.n:
                n = 0
                epoch_num = epoch_num + 1
                cerr = 0
                for cn in range(self.n):
                    if cn % 10 == 0:
                        yhat = 1 / (1 + np.exp(-np.dot(self.features[cn], w)))
                        cerr = cerr - self.labels[cn] * np.log(yhat) - (1 - self.labels[cn]) * np.log(1 - yhat)
                if cerr < epsilon:
                    break
        

    def logistic_testing(self, testX):
        """TestX is a numpy array, which is pre-determined in this hw, referring to
        test_data in auto-testing part. (Friendly remainder: test_data is a panda dataframe)
        Use your trained weight and bias to compute the predicted y values,
        Predicted y values should be 0 or 1. return the numpy array in shape n*1"""
        n = np.shape(testX)[0]
        y = np.empty((n, 1))
        data = np.c_[ testX, np.ones(n) ]

        for i in range(n):
            for k in range(30):
                data[i,k] = (data[i,k] - self.mean[k]) / self.stdev[k]

        for i in range(n):
            for k in range(31):
                yhat = 1 / (1 + np.exp(-np.dot(data[i], self.w)))
                if yhat > 0.5:
                    y[i] = 1
                else:
                    y[i] = 0
        return y


    def svm_training(self, gamma, C):
        """Training process of the support vector machine will happen here. We will always use Gaussian (rbf) kernel,
        and the grid search parameters are gamma and C, which are both python lists with min and max range values.
        You are allowed to use build-in grid search for SVM. Consulting documentation for specific usage.
        As softmargin SVM can handle nonlinear boundaries and outliers much better than simple logistic regression,
        we do not perform 3-fold validation here (just one training run with 90-10 training-validation split).
        Furthmore, you are allowed to use SVM's built-in grid search method.
        This function will be a "wrapper" around sklearn.svm.SVC with all other parameters take the default values.
        Please consult sklearn.svm.SVC documents to see how to use its "fit" and "predict" functions. """
        g_vals = np.linspace(gamma[0], gamma[1], 10)
        C_vals = np.linspace(C[0], C[1], 10)

        parameters = {'C': C_vals, 'gamma': g_vals}
        svc = SVC() # kwargs={'gamma':gamma, 'C':C})
        self.grid = GridSearchCV(estimator=svc, param_grid=parameters)
        self.grid.fit(self.features, self.labels)
        
    def svm_testing(self, testX):
        """TestX is a numpy array, which is pre-determined in this hw, referring to
        test_data in auto-testing part. (Friendly remainder: test_data is a panda dataframe)
        Use your trained SVM to return the numpy array in shape n*1, predicted y values should be 0 or 1"""
        n = np.shape(testX)[0]
        data = np.c_[ testX, np.ones(n) ]

        for i in range(n):
            for k in range(30):
                data[i,k] = (data[i,k] - self.mean[k]) / self.stdev[k]

        y = self.grid.predict(data)
        return y

""" Training Process: You only need to modify nepoch, epsilon of logistic training method.
Please donot modify anything for SVM training function. Please don't add your own svm_testing
or logistic_testing function when you submitting this HW. This is for auto-testing """
dataset = load_breast_cancer(as_frame=True)
"""Dataset is divided into 90% and 10%, 90% for you to perform k-fold validation and 10% for
auto-tester to validate your performance. Please Donot change random_state, which will generate
difffernt partitions. We want to ensure fair competition among all students"""
train_data = dataset['data'].sample(frac=0.9, random_state=0) # random state is a seed value
train_target = dataset['target'].sample(frac=0.9, random_state=0) # random state is a seed value
test_data = dataset['data'].drop(train_data.index)
test_target = dataset['target'].drop(train_target.index)

model = BinaryClassifier(train_data, train_target)
"""Only performance of logistic regression will be used for competition. However, you must implement SVM training
and Testing, generates reasonable results, above 90%"""
# Compute the time to do grid search on training logistic
logistic_start = time.time()
model.logistic_training([10**-10, 10], [10e-10, 1e10], 100, 0)
logistic_end = time.time()
# Compute the time to do grid search on training SVM
svm_start = time.time()
model.svm_training([1e-9, 1000], [0.01, 1e10])
svm_end = time.time()

# end of skeleton code, this code for testing

print('TRAINING FINISHED:\ntime:\t%s\nalpha:\t%s\nlambda:\t%s\nerror:\t%s\nw:\n%s' % ((logistic_end - logistic_start), model.a, model.l, model.error, model.w))
prediction = model.logistic_testing(test_data.to_numpy())
actual = test_target.to_numpy()
correct = 0
size = test_target.shape[0]
for i in range(size):
    if prediction[i] == actual[i]:
        correct = correct + 1
print('TESTING FINISHED:\nlogistic regression model:\ncorrect:\t%s / %s\npercent:\t%s' % (correct, size, correct / size))
prediction = model.svm_testing(test_data.to_numpy())
correct = 0
for i in range(size):
    if prediction[i] == actual[i]:
        correct = correct + 1
print('svm.SVC model:\ncorrect:\t%s / %s\npercent:\t%s' % (correct, size, correct / size))
    