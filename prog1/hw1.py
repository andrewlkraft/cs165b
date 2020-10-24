import numpy as np, time, matplotlib.pyplot as plt

class SGDSolver():
    def __init__(self, path):
        """ load input dataset specified in path and split data into train and validation. Hint: you can
        store both training and testing features and target vector as class variables """
        tmp = []
        try:
            with open(path) as f:
                f.readline() # read past title line
                for line in f:
                    tmp.append([float(x) for x in line.split(',')[1:]])
        except Exception as e:
            print('Could not open file:\n%s' % e)
            exit(0)
        self.rng = np.random.default_rng()
        self.rng.shuffle(tmp)
        self.x = np.array(tmp)[:,:7]
        self.y = np.array(tmp)[:,7]
        self.w = np.empty(7)
        self.b = 0
        self.mse = np.inf

    def training(self, alpha, lam, nepoch, epsilon):
        """ Training process of linear regression will happen here. User will provide
        learning rate alpha, regularization term lam, specific number of training epoches,
        and a variable epsilon to specify pre-mature end condition,
        ex. if error < epsilon, training stops. Hint: You can store both weight and
        bias as class variables, so other functions can directly use them """
        kxval = 6
        
        tmp_l = lam[0]
        tmp_a = alpha[0]
        lambdas = []
        alphas = []
        while tmp_l <= lam[1]:
            lambdas.append(tmp_l)
            tmp_l = tmp_l * 10
        while tmp_a <= alpha[1]:
            alphas.append(tmp_a)
            tmp_a = tmp_a * 10
        n = len(self.x)

        # 2 nested loops to perform the grid search of alpha and lambda
        for a in alphas:
            for l in lambdas:
                for xval in range(kxval):
                    b = self.rng.random()
                    w = self.rng.random(7)
                    # perform SGD number of times = nepoch
                    mse = np.inf
                    for iteration in range(nepoch):
                        # each iteration calculates b_grad and w_grad once
                        b_grad = 0
                        w_grad = np.zeros(7)

                        for index in range(n):
                            # only use sample in gradient calculation if its index is not equal to val mod self.k, ie only if in training set
                            if index % kxval != xval:
                                # calculate the part of the gradient for each entry, and add it to the gradient overall
                                tmp_sum = 2 / n * (b + w @ self.x[index] - self.y[index])
                                for k in range(7):
                                    w_grad[k] = w_grad[k] + tmp_sum * self.x[index,k] + l * w[k] / n
                                b_grad = b_grad + tmp_sum
                        # perform SGD
                        tmp_b = b - a * b_grad
                        tmp_w = w - a / np.sqrt(iteration + 1) * w_grad

                        # evaluate mse
                        tmp_mse = 0
                        for index in range(n):
                            tmp_mse = tmp_mse + (tmp_b + tmp_w @ self.x[index] - self.y[index])**2 / n
                        if tmp_mse > mse:
                            break

                        # if mse went down (or stayed level) since last epoch, keep going
                        mse = tmp_mse
                        b = tmp_b
                        w = tmp_w

                        # if mse is low enough, break because of diminishing returns
                        if mse < epsilon:
                            break
                if mse < self.mse:
                    self.mse = mse
                    self.b = b
                    self.w = w
                

    def testing(self, testX):
        """ predict the Y value based on testing data input and ground truth data """
        n = range(testX)
        testY = np.array((n,1))
        for index in range(n):
            testY[index,0] = testX[index] @ self.w + self.b
        return testY

""" Training Process: You only need to modify nepoch, epsilon of training method,
this is for autograding """
model = SGDSolver('tests/train.csv')
# Compute the time to do grid search on training
start = time.time()
model.training([10**-10, 10], [1, 1e10], 300, 0.01)
end = time.time()
# print('---RESULTS---\ntraining time:\t%s\nMODEL:\ntraining & validation mse:\t%s\nb:\t%s\nw:\n%s' % (end - start, model.mse, model.b, model.w))