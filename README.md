# svr-smo-scratch-python
Support vector regression with SMO training from scratch using Python
This implementation of support vector regression uses the sequential minimal optimization (SMO) algorithm for training (as opposed to quadratic programming or stochastic sub-gradient descent).

The implementation uses a simple, single weights vector named alpha, instead of the common dual vectors, usually named alpha and alpha* approach.

The implementation uses a hard-wired radial basis function (RBF) as the kernel function, as opposed to allowing different functions like the polynomial kernel or the linear kernel.

SVR was popular for a short time in the late 1990s and early 2000s, until people discovered that the closely related kernel ridge regression (KRR) is superior to SVR in nearly every way.

SVR is more difficult to implement than SVR. SVR parameters (gamma, epsilon, C, max iterations, tolerance) are much more difficult to tune than KRR parameters. SVR models typically don't perform as well as KRR models in terms of prediction accuracy.

All that said, there are some problem domains where SVR is still used. And SVR is extremely interesting from a mathematical point of view.

## Usage

One of the reasons that SVR parameters are difficult to tune is that they are intertwined. And slight changes can effectively move training vectors in and out of the epsilon tube, which creates large changes in the model.

The demo implementation loosely follows the scikit-learn SVR module API. Example calling code:

    # X is a numpy matrix of predictors
    # y is a numpy vector of target values
    model = KernelSVR(gamma=0.30, epsilon=0.001, C=1.0, max_iter=100, tol=1.0e-5, seed=0)
    model.fit(X, y)

    print("Predicting first training item")
    x = X[0].reshape(1, -1)
    pred_y = model.predict(x)[0]
    print("Predicted y = %0.4f " % pred_y)

    idxs = model.supp_vec_idxs()
    print("The indexes of the support vectors are: ")
    print(idxs)  # training items associated with near-zero alpha wts

The gamma parameter controls the RBF kernel function. Increasing gamma shrinks the radius of influence of individual data points. This causes the model to give more weight to points that are very close to each other.

The epsilon parameter controls which items are ignored during training. Increasing epsilon tends to create fewer support vectors (those with non-zero alpha values).

The C parameter controls regularization, to prevent alpha weights from becoming large. Increasing C increases the penalty for points falling outside the epsilon tube. This forces the model to fit the training data more strictly, which increases accuracy but creates an increased risk of model overfitting.

The max_iter parameter sets the maximum consecutive number of times the SMO algorithm iterates without finding an improvement in alpha.

The tol parameter sets a tolerance for the KTT conditions. Increasing tol allows more updates to occur. 

In practice, tuning SVR parameters is often extremely difficult.
