import numpy as np, time, matplotlib.pyplot as plt

TAKE_AVG = True
KXVAL = 8
EPOCHS = 300
EPSILON = 0.0001

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
        
        tmp_l = 1000
        tmp_a = 1e-10
        lambdas = [tmp_l]
        alphas = [tmp_a]

        while tmp_l < 1e6:
            tmp_l = tmp_l * 3
            lambdas.append(tmp_l)
        
        while tmp_a < 1e-6:
            tmp_a = tmp_a * 3
            alphas.append(tmp_a)

        n = len(self.x)

        # 2 nested loops to perform the grid search of alpha and lambda
        for a in alphas:
            for l in lambdas:

                b_avg = 0
                w_avg = np.zeros(7)
                if TAKE_AVG:
                    mse_avg = 0
                else:
                    mse_avg = np.inf
                
                for xval in range(KXVAL):
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
                            if index % KXVAL != xval:
                                # calculate the part of the gradient for each entry, and add it to the gradient overall
                                inner = 2 / n * (b + np.dot(w, self.x[index]) - self.y[index])

                                w_grad = inner * self.x[index] + l * w / n
                                b_grad = inner

                                tmp_b = b - a / np.sqrt(iteration + 1) * b_grad
                                tmp_w = w - a / np.sqrt(iteration + 1) * w_grad
                        # perform SGD



                        # evaluate mse
                        valid = 0
                        training = 0
                        for index in range(n):
                            if index % KXVAL == xval:
                                valid = valid + (tmp_b + np.dot(tmp_w, self.x[index]) - self.y[index])**2 / n
                            else:
                                training = training + (tmp_b + np.dot(tmp_w, self.x[index]) - self.y[index])**2 / n

                        if valid > mse:
                            break

                        # if mse went down (or stayed level) since last epoch, keep going
                        mse = valid
                        b = tmp_b
                        w = tmp_w

                        
                        # if mse is low enough, break because of diminishing returns
                        if mse < epsilon:
                            break
                    
                    # take the results of model building from this training and validation test and add to average
                    # OR select if less than running minimum, if TAKE_AVG is set to False
                    
                    if TAKE_AVG == True:        
                        b_avg = b_avg + b / KXVAL
                        w_avg = w_avg + w / KXVAL
                        mse_avg = mse_avg + mse / KXVAL
                    else:
                        if mse < mse_avg:
                            mse_avg = mse
                            b_avg = b
                            w_avg = w
                if mse_avg < self.mse:
                    self.mse = mse_avg
                    self.b = b_avg
                    self.w = w_avg

    def testing(self, testX):
        """ predict the Y value based on testing data input and ground truth data """
        n = len(testX)
        testY = np.zeros((n,1))
        for index in range(n):
            testY[index] = testX[index] @ self.w + self.b
        return testY

""" Training Process: You only need to modify nepoch, epsilon of training method,
this is for autograding """
model = SGDSolver('tests/train.csv')
# Compute the time to do grid search on training
start = time.time()
model.training([10**-10, 10], [1, 1e10], EPOCHS, EPSILON)
end = time.time()
print('---COMPLETE---\nTRAINING:\ntraining time:\t%s\nepochs:\t%s\nepsilon:\t%s\nMODEL:\nmse:\t%s\nkxval:\t%s\navging:\t%s\nb:\t%s\nw:\n%s' % (end - start, EPOCHS, EPSILON, model.mse, KXVAL, TAKE_AVG, model.b, model.w))