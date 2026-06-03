# svr_smo.py

# kernel support vector regression with SMO training.
# hard-wired RBF kernel function.
# uses single signed alpha (weights), not alpha and alpha*.

# I indent using 2 spaces instead of the usual 4 spaces --
# just a personal preference.

import numpy as np

# -----------------------------------------------------------

np.set_printoptions(precision=4, suppress=True,
  floatmode='fixed', linewidth=120)

# -----------------------------------------------------------
# external eval functions: accuracy(), mse(), r2_score()
# -----------------------------------------------------------

def accuracy(model, data_X, data_y, pct_close):
  n = len(data_X)
  n_correct = 0; n_wrong = 0
  for i in range(n):
    x = data_X[i].reshape(1,-1)
    y = data_y[i]
    pred_y = model.predict(x)[0]
    if np.abs(y - pred_y) < np.abs(y * pct_close):
      n_correct += 1
    else: 
      n_wrong += 1
  return n_correct / (n_correct + n_wrong)

# -----------------------------------------------------------

def mse(model, data_X, data_y):
  n = len(data_X)
  sum = 0.0
  for i in range(n):
    x = data_X[i].reshape(1,-1)
    y = data_y[i]
    pred_y = model.predict(x)[0]
    diff = pred_y - y
    sum += diff * diff
  return sum /n

# -----------------------------------------------------------

def r2_score(model, data_X, data_y):
  # coefficient of determination == scikit score()
  ss_res = 0.0
  ss_tot = 0.0
  n = len(data_X)
  mean_y = np.mean(data_y)
  for i in range(n):
    x = data_X[i].reshape(1,-1)
    y = data_y[i]
    pred_y = model.predict(x)[0]
    ss_res += (y - pred_y) * (y - pred_y)
    ss_tot += (y - mean_y) * (y - mean_y)
  result = 1.0 - (ss_res / ss_tot)
  return result

# ===========================================================

class KernelSVR:
  def __init__(self, gamma=1.0, epsilon=0.01, C=1.0, 
    max_iter=100, tol=1.0e-3, seed=0):
    self.gamma = gamma
    self.epsilon = epsilon
    self.C = C

    self.max_iter = max_iter
    self.tol = tol

    self.alpha = None
    self.b = 0.0
    self.supp_X = None

    self.rnd = np.random.RandomState(seed)

  # ---------------------------------------------------------

  def kernel_matrix(self, X1, X2):
    # Computes pairwise RBF kernel matrix or vectors
    sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
    np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-self.gamma * sq_dists)

  # ---------------------------------------------------------

  def fit(self, X, y):
    self.supp_X = X  # needed by predict()
    self.supp_y = y  # not used this version
    n = X.shape[0]
    self.alpha = np.zeros(n)  # single signed version
    self.b = np.mean(y)
    
    # kernel matrix for faster predictions
    K = self.kernel_matrix(X, X)

    n_passes = 0  # number times no update
    while n_passes < self.max_iter:
      num_changed_alphas = 0
      
      for i in range(n):  # each item i in order
        pred_i = np.dot(self.alpha, K[:, i]) + self.b
        err_i = pred_i - y[i]

        # check KKT conditions within a tolerance
        if ((err_i > self.epsilon - self.tol and \
             self.alpha[i] > -self.C) or \
            (err_i < -self.epsilon + self.tol and \
             self.alpha[i] < self.C)):
          
          # random second index j, not equal i
          j = i
          while j == i:
            j = self.rnd.randint(0,n)
                    
          pred_j = np.dot(self.alpha, K[:, j]) + self.b
          err_j = pred_j - y[j]

          # save previous weight values
          alpha_i_old, alpha_j_old = \
          self.alpha[i], self.alpha[j]

          # upper and lower bounds for single signed alpha
          sa = alpha_i_old + alpha_j_old  # sum of alphas
          L = max(-self.C, sa - self.C)
          H = min(self.C, sa + self.C)
          
          if L == H:
            continue

          # kernel second derivative step denominator
          eta = K[i, i] + K[j, j] - (2 * K[i, j])
          if eta <= 0.0:
            continue

          # update weight j
          self.alpha[j] = alpha_j_old + (err_i - err_j) / eta
          self.alpha[j] = np.clip(self.alpha[j], L, H)

          if abs(self.alpha[j] - alpha_j_old) < 1e-5:
            continue

          # update weight i to keep balance sum invariant
          self.alpha[i] = sa - self.alpha[j]

          # update bias
          b1 = self.b - err_i - (self.alpha[i] - \
            alpha_i_old) * K[i, i] - (self.alpha[j] - \
            alpha_j_old) * K[j, i]
          b2 = self.b - err_j - (self.alpha[i] - \
            alpha_i_old) * K[i, j] - (self.alpha[j] - \
            alpha_j_old) * K[j, j]

          if -self.C < self.alpha[i] < self.C:
            self.b = b1
          elif -self.C < self.alpha[j] < self.C:
            self.b = b2
          else:
            self.b = (b1 + b2) / 2.0

          num_changed_alphas += 1

        else:
          pass  # no update to item i

      if num_changed_alphas == 0:
        n_passes += 1
      else:
        n_passes = 0  # reset

    # at this point, you could prune away items in supp_X
    # that are associated with weights that are nearly
    # zero, which are the support vectors.

    return  # all done

  # ---------------------------------------------------------

  def predict(self, X):
    K_test = self.kernel_matrix(X, self.supp_X)
    return np.dot(K_test, self.alpha) + self.b

  # ---------------------------------------------------------

  def get_supp_idxs(self):
    result = []
    for i in range(len(self.alpha)):
      # a nearly-zero wt is associated with a supp vector
      if np.abs(self.alpha[i]) > 1.0e-5:
        result.append(i)
    return result

# ===========================================================

def main():
  print("\nBegin scratch Python SVR using SMO training ")

  ## quick sanity check
  # np.random.seed(0)
  # n_samples = 40; n_features = 4
  # X = np.random.randn(n_samples, n_features)
  # weights = np.array([0.2, -0.5, 0.3,  0.1])
  # bias = 0.45
  # y = X @ weights + bias + np.random.randn(n_samples)

  # print("\nX = "); print(X[0:3,:]); print(" . . . ")
  # print("\ny = "); print(y[0:3], end=""); print(" . . . ")

  # model = KernelSVR(gamma=0.50, epsilon=0.01, C=1.0, 
  #   max_iter=20, tol=1.0e-5)
  # model.fit(X, y)

  # MSE = mse(model, X, y)
  # print("\nModel MSE = %0.4f " % MSE)

  print("\nLoading synthetic train (200) and test (40) data")
  train_Xy = np.loadtxt(".\\Data\\synthetic_train_200.txt",
    usecols=[0,1,2,3,4,5], delimiter=",")
  train_X = train_Xy[:,[0,1,2,3,4]]
  train_y = train_Xy[:,5]

  test_Xy = np.loadtxt(".\\Data\\synthetic_test_40.txt",
    usecols=[0,1,2,3,4,5], delimiter=",")
  test_X = test_Xy[:,[0,1,2,3,4]]
  test_y = test_Xy[:,5]
  print("Done ")

  print("\nFirst three train X: ")
  for i in range(3):
    print(train_X[i])
  print("\nFirst three train y: ")
  for i in range(3):
    print("%0.4f " % train_y[i])

  # ** SCIKIT results **
  # Setting gamma = 0.3000
  # Setting C = 1.0
  # Setting epsilon = 0.0010
  # Number model support vectors: [184]
  # Model bias: 0.4063
  # Train accuracy (0.10) = 0.9850
  # Test accuracy (0.10) = 0.9500  
  # Train MSE = 0.0000
  # Test MSE = 0.0002

  # create and train model
  print("\nCreating SVR-SMO model ")
  gamma = 0.30
  epsilon = 0.001
  C = 1.0
  max_iter = 100  # max number iter with no improve
  tol = 1.0e-5

  print("Setting gamma = %0.4f " % gamma)
  print("Setting C = %0.2f " % C)
  print("Setting epsilon = %0.6f " % epsilon)
  print("Setting max_iter = " + str(max_iter))
  print("Setting tol = %0.6f " % tol)

  print("\nCreating and training SVR model using SMO ")

  model = KernelSVR(gamma=gamma, epsilon=epsilon, C=C, 
    max_iter=max_iter, tol=tol)

  model.fit(train_X, train_y)
  print("Done ")

  print("\nModel alpha: ")
  print(model.alpha)
  # print("\nModel alpha*: ") # uses single alpha
  # print(model.alpha_star)

  supp_vec_idxs = model.get_supp_idxs()
  print("Number support vectors = " + \
    str(len(supp_vec_idxs)))

  acc_train = accuracy(model, train_X, train_y, 0.10)
  print("\nTrain accuracy (0.10) = %0.4f" % acc_train)
  acc_test = accuracy(model, test_X, test_y, 0.10)
  print("Test accuracy (0.10) = %0.4f" % acc_test)

  mse_train = mse(model, train_X, train_y)
  print("\nTrain MSE = %0.4f" % mse_train)
  mse_test = mse(model, test_X, test_y)
  print("Test MSE = %0.4f" % mse_test)

  # r2_train = r2_score(model, train_X, train_y)
  # print("\nTrain R2 = %0.4f" % r2_train)
  # r2_test = r2_score(model, test_X, test_y)
  # print("Test R2 = %0.4f" % r2_test)

  print("\nPredicting for train_X[0] ")
  x = train_X[0].reshape(1,-1)
  pred_y = model.predict(x)[0]
  print("Predicted y = %0.4f " % pred_y)

  print("\nEnd demo ")

# -----------------------------------------------------------

if __name__ == "__main__":
  main()
