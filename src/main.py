from model import *
from data import load_data

X_train, Y_train, X_test, Y_test = load_data()

epochs      = 100
batch_size  = 100
lr          = 0.001
n_batches   = 60000 // batch_size   
t_acc       = 0

input_size  = 784
hidden1     = 128
hidden2     = 64
output_size = 10

W1, b1, W2, b2, W3, b3 = init_params(input_size, hidden1, hidden2, output_size)
params       = (W1, b1, W2, b2, W3, b3)
adam_buffers = init_adam(W1, b1, W2, b2, W3, b3)
t            = 0

print("\nTraining started...\n")

for epoch in range(1, epochs+1):

    idx      = np.random.permutation(60000)
    X_shuf   = X_train[idx]
    Y_shuf   = Y_train[idx]

    epoch_loss = 0.0

    for i in range(n_batches):

        X_batch = X_shuf[i*batch_size : (i+1)*batch_size]  # (100, 784)
        Y_batch = Y_shuf[i*batch_size : (i+1)*batch_size]  # (100, 10)

        W1, b1, W2, b2, W3, b3 = params

        Y_hat, cache = forward(X_batch, W1, b1, W2, b2, W3, b3)

        loss        = compute_loss(Y_hat, Y_batch)
        epoch_loss += loss

        grads = backward(Y_batch, cache, W2, W3)

        t += 1
        params, adam_buffers = adam_update(params, grads, adam_buffers, t, lr=lr)

    W1, b1, W2, b2, W3, b3 = params

    avg_loss    = epoch_loss / n_batches
    train_acc   = accuracy(X_train, Y_train, W1, b1, W2, b2, W3, b3)
    t_acc      += train_acc / epochs

    if(epoch % 10 == 0):
        print(f"Epoch {epoch:>3} | Loss: {avg_loss:.4f} | Training Acc: {train_acc*100:.2f}%")

final_acc = accuracy(X_test, Y_test, W1, b1, W2, b2, W3, b3)
print(f"Training Acc Average: {t_acc*100:.2f}% \nTest Accuracy: {final_acc*100:.2f}%")