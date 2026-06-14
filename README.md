# mnist-neural-network

A fully functional neural network built **from scratch using only NumPy**, trained on the MNIST handwritten digit dataset. No PyTorch. No TensorFlow. No autograd. Every single forward pass, backpropagation step, and weight update is implemented manually.

**Final Test Accuracy : 97.97% on 10,000 unseen images.**

---

## What This Is

Most people learn neural networks by calling `model.fit()`. This project does the opposite — every line of math is written by hand, from He initialization to the Adam optimizer's bias correction. The goal was to understand what actually happens inside a neural network before using libraries that abstract it away.

---

## Architecture

```
Input (784) --> Hidden Layer 1 (128) --> Hidden Layer 2 (64) --> Output (10)
                     ReLU                      ReLU                Softmax
```

| Layer | Neurons | Activation | Parameters |
|---|---|---|---|
| Input | 784 | — | 0 |
| Hidden 1 | 128 | ReLU | 100,480 |
| Hidden 2 | 64 | ReLU | 8,256 |
| Output | 10 | Softmax | 650 |
| **Total** | | | **109,386** |

---

## Results

| Metric | Value |
|---|---|
| Training Accuracy | 99.95% |
| Test Accuracy | 97.97% |
| Epochs | 100 |
| Batch Size | 100 |
| Optimizer | Adam |
| Learning Rate | 0.001 |

---

## What is Implemented From Scratch

### Forward Pass
- Matrix multiplication layer by layer : `Z = X @ W + b`
- ReLU activation : `max(0, Z)`
- Numerically stable Softmax : subtract max before exp to prevent overflow

### Loss
- Categorical Cross-Entropy : `L = -log(ŷ_correct)`
- Batch averaged cost : `C = -sum(Y * log(Ŷ)) / m`
- Log clipping to prevent `log(0)` → NaN

### Backpropagation
- Combined Softmax + Cross-Entropy gradient : `dZ3 = (Ŷ - Y) / m`
- Full chain rule through all 3 layers
- ReLU gradient gate : `dZ = dA * (Z > 0)`
- Weight gradients via matrix transpose : `dW = A_prev.T @ dZ`
- Bias gradients via batch sum : `db = sum(dZ, axis=0)`

### Adam Optimizer
- First moment (momentum) : `m = β₁·m + (1-β₁)·dW`
- Second moment (variance) : `v = β₂·v + (1-β₂)·dW²`
- Bias correction : `m̂ = m / (1 - β₁ᵗ)`, `v̂ = v / (1 - β₂ᵗ)`
- Adaptive update : `W = W - (α / (√v̂ + ε)) · m̂`

### Weight Initialization
- He initialization for ReLU networks : `W = randn * sqrt(2 / n_in)`
- Prevents vanishing and exploding gradients from the start

### Training Loop
- Mini-batch gradient descent (batch size = 100)
- Dataset shuffle at every epoch
- Validation accuracy tracked every 10 epochs

---

## Project Structure

```
mnist-neural-network/
│
├── imports.py      # numpy and sklearn imports
├── data.py         # load, normalize, one-hot encode, train/test split
├── network.py      # all neural network functions
└── main.py         # hyperparameters, training loop, evaluation
```

### File Responsibilities

**imports.py**
- numpy
- sklearn fetch_openml

**data.py**
- loads MNIST via sklearn (70,000 images)
- normalizes pixel values to [0, 1]
- one-hot encodes labels to shape (m, 10)
- splits into 60,000 train / 10,000 test

**network.py**
- `init_params()` — He initialized weights, zero biases
- `init_adam()` — 12 zero buffers for Adam
- `relu()`, `relu_gate()`, `softmax()` — activation functions
- `forward()` — full forward pass with cache
- `compute_loss()` — cross-entropy with clip
- `backward()` — full backpropagation
- `adam_update()` — complete Adam step
- `accuracy()` — argmax comparison on any split

**main.py**
- hyperparameter definitions
- training loop (epochs × batches)
- per-epoch loss and accuracy reporting
- final test evaluation

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/HackerX-47/mnist-neural-network.git
cd mnist-neural-network
```

### 2. Install dependencies
```bash
pip install numpy scikit-learn
```

### 3. Run
```bash
python main.py
```

### First Run Note
sklearn downloads MNIST (~55MB) on first run and caches it locally.
This takes 30-60 seconds. All subsequent runs load instantly.

### Expected Output
```
Loading MNIST dataset...
Done.
X_train : (60000, 784)
Y_train : (60000, 10)
X_test  : (10000, 784)
Y_test  : (10000, 10)

Training started...

Epoch   1 | Loss: 2.1847 | Train: 41.23%
Epoch   2 | Loss: 1.4923 | Train: 68.91%
...
Epoch  10 | Loss: 0.4231 | Train: 88.64% | Val: 88.21%
...
Epoch 100 | Loss: 0.0812 | Train: 99.95% | Val: 97.97%

Final Test Accuracy: 97.97%
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Input size | 784 | 28×28 pixels flattened |
| Hidden 1 | 128 | tunable |
| Hidden 2 | 64 | tunable |
| Output size | 10 | digits 0-9 |
| Epochs | 100 | |
| Batch size | 100 | 600 batches per epoch |
| Learning rate | 0.001 | Adam default |
| β₁ | 0.9 | Adam momentum |
| β₂ | 0.999 | Adam variance |
| ε | 1e-8 | Adam stability |

---

## Key Concepts Implemented

**He Initialization**
Weights drawn from N(0, sqrt(2/n_in)). Specifically designed for ReLU networks to keep activation variance stable across layers, preventing vanishing and exploding gradients.

**ReLU Gate in Backprop**
During backpropagation, the ReLU derivative acts as a binary gate — open (gradient passes through unchanged) for neurons that were active during the forward pass, closed (gradient blocked) for neurons that output zero. This is why dead neurons are catastrophic — they block gradient flow for every layer behind them.

**Softmax + Cross-Entropy Gradient Cancellation**
Differentiating cross-entropy through softmax separately produces complex terms. Combined, they cancel to the remarkably clean expression `dZ = Ŷ - Y`. This is not a coincidence — cross-entropy was designed to pair with softmax to produce this result.

**Adam Bias Correction**
Both Adam moment buffers initialize at zero, causing severe underestimation of true gradients in early batches. Dividing by `(1 - βᵗ)` corrects this — the correction is large at t=1 and self-extinguishes toward zero as training progresses.

**Mini-Batch Averaging**
The `/m` in `dZ = (Ŷ - Y) / m` normalizes gradients per image regardless of batch size. Without it, changing batch size would require retuning the learning rate.

---

## Dependencies

```
numpy
scikit-learn
```

No deep learning frameworks. No automatic differentiation. Just math.

---

## Author

**HackerX-47**
https://github.com/HackerX-47