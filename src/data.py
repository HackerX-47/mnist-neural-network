from imports import *

def load_data(): 
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X     = mnist.data                  # (70000, 784)
    y     = mnist.target.astype(int)    # (70000,)

    X = X / 255.0                       # range [0, 1]

    def one_hot(y, classes=10):
        m = y.shape[0]
        Y = np.zeros((m, classes))
        Y[np.arange(m), y] = 1
        return Y


    Y = one_hot(y)                      # (70000, 10)

    X_train = X[:60000]                 # (60000, 784)
    Y_train = Y[:60000]                 # (60000, 10)
    X_test  = X[60000:]                 # (10000, 784)
    Y_test  = Y[60000:]                 # (10000, 10)

    print(f"X_train : {X_train.shape}")
    print(f"Y_train : {Y_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"Y_test  : {Y_test.shape}")
    print(f"Range   : {X_train.min()} to {X_train.max()}")

    return X_train, Y_train, X_test, Y_test