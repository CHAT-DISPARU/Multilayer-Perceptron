# Multilayer Perceptron : breast cancer detection

A neural network **written from scratch with NumPy** (without TensorFlow, PyTorch or scikit-learn) that learns to classify a breast tumor as **malignant (M)** or **benign (B)** from measurements made on cell images.

This README has two goals : explain how to use the project, and above all explain **why it works** : the math, the algorithms and the choices made.

---

## Table of contents

1. [The subject](#1-the-subject)
2. [The data](#2-the-data)
3. [Project structure](#3-project-structure)
4. [Installation and usage](#4-installation-and-usage)
5. [The basics of a neural network](#5-the-basics-of-a-neural-network)
6. [The forward pass (forward)](#6-the-forward-pass-forward)
7. [Activation functions](#7-activation-functions)
8. [Measuring the error : the cost function](#8-measuring-the-error--the-cost-function)
9. [Backpropagation (backpropagation)](#9-backpropagation-backpropagation)
10. [The optimizers : SGD and Adam](#10-the-optimizers--sgd-and-adam)
11. [Weight initialization](#11-weight-initialization)
12. [Data standardization](#12-data-standardization)
13. [The training loop](#13-the-training-loop)
14. [Early stopping](#14-early-stopping)
15. [Evaluation metrics](#15-evaluation-metrics)
16. [Reading the training output](#16-reading-the-training-output)
17. [Glossary](#17-glossary)

---

## 1. The subject

This project is a classic machine learning exercise : **implement a multilayer perceptron yourself** (*multilayer perceptron*, MLP) and use it on a real medical problem.

### The medical context

Breast cancer is one of the most common cancers. When an abnormality is detected (palpation, mammography), the doctor often performs a **fine needle aspiration** : a few cells are taken from the suspicious mass. These cells are observed under a microscope, and a digitized image is analyzed to measure the shape of the cell nuclei : their size, their regularity, the sharpness of their contour, etc.

The goal is to know whether the tumor is :

- **benign (B)** : it does not spread, it is generally not very dangerous ;
- **malignant (M)** : it is a cancer, which can invade neighboring tissues.

A neural network can learn, from hundreds of already diagnosed cases, which combinations of measurements are associated with a malignant tumor. It is a **supervised binary classification** problem :

- *classification* : the answer is a category (M or B), not a number ;
- *binary* : there are two categories ;
- *supervised* : we learn from examples whose correct answer is known.


### What the program does

1. `split.py` splits the dataset into a **training** set and a **validation/prediction** set.
2. `training.py` trains the network, displays the learning curves and saves the model.
3. `predict.py` reloads the model and predicts the diagnosis of new data.

---

## 2. The data

The dataset is the **Wisconsin Diagnostic Breast Cancer (WDBC)** : 569 patients, including about 357 benign cases and 212 malignant cases.

Each row of the CSV contains 32 columns :

| Column | Content |
|---|---|
| 1 | `ID` : patient identifier (not used for learning) |
| 2 | `Diagnosis` : `M` (malignant) or `B` (benign) : this is the **target** |
| 3 to 32 | **30 numerical characteristics** (the *features*) |

The 30 characteristics come from **10 measurements** made on each cell nucleus :

| # | Measurement | Meaning |
|---|---|---|
| a | radius | mean of the distances from the center to the points of the contour |
| b | texture | standard deviation of the gray levels |
| c | perimeter | perimeter |
| d | area | area |
| e | smoothness | local variation of the length of the radii |
| f | compactness | perimeter² / area − 1 |
| g | concavity | severity of the dips in the contour |
| h | concave points | number of dips in the contour |
| i | symmetry | symmetry |
| j | fractal dimension | « coastline approximation » − 1 |

For each of these 10 measurements, we have three versions : the **mean** (`_mean`), the **standard error** (`_se`) and the **worst value** (`_worst`, mean of the three largest values). That makes 10 × 3 = **30 features**.

Intuitively, cancerous cells tend to be larger, more irregular, with a more jagged contour. The network must discover by itself which combination of these signals is the most discriminating.

### The data is not balanced

About 63 % of the cases are benign. A « model » that always answered *benign* would therefore already have ~63 % correct answers without learning anything. That is why we don't look only at the accuracy, but also at the precision, the recall and the F1 (see [section 15](#15-evaluation-metrics)).

---

## 3. Project structure

```
.
├── split.py              # splits data.csv into train / validation
├── training.py           # trains the network, plots the curves, saves
├── predict.py            # loads a model and predicts on a CSV
├── mlp/
│   ├── __init__.py
│   ├── csv_reader.py     # strict reading and validation of the CSV
│   ├── activations.py    # relu, sigmoid, tanh, softmax + derivatives
│   ├── layer.py          # a dense layer : forward / backward
│   ├── network.py        # stacks the layers, fit, save, load
│   ├── losses.py         # binary cross-entropy, accuracy, precision, recall, F1
│   └── optimiseur.py     # SGD and Adam
└── datasets/             # created by split.py
    ├── data_train.csv
    └── data_predict.csv
```

The project is **modular** : the architecture of the network (number and size of the layers), the optimizer, the activation function and the hyperparameters are set on the command line, without touching the code.

---

## 4. Installation and usage

### Dependencies

```bash
pip install numpy matplotlib
```

### Step 1 : split the data

```bash
python split.py data.csv 0.8            # 80 % train, 20 % validation, reproducible split
python split.py data.csv 0.8 --rand     # random split, different at each run
```

This creates `datasets/data_train.csv` and `datasets/data_predict.csv`. Without `--rand`, the seed is fixed at 42 : we always get the same split.

### Step 2 : train

```bash
python training.py --train datasets/data_train.csv --val datasets/data_predict.csv
```

Complete example :

```bash
python training.py --train datasets/data_train.csv --val datasets/data_predict.csv \
    --layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
```

| Option | Default | Role |
|---|---|---|
| `--train` | (required) | training CSV |
| `--val` | (required) | validation CSV |
| `--layer` | `24 24` | sizes of the hidden layers (at least 2) |
| `--epochs` | `70` | number of complete passes over the data |
| `--batch_size` | `16` | size of the mini-batches |
| `--lr` / `--learning_rate` | `0.001` (Adam), `0.01` (SGD) | learning step |
| `--optimizer` | `adam` | `adam` or `sgd` |
| `--patience` | `0` (disabled) | early stopping : stop after N epochs without improvement |
| `--save_path` | `saved_model` | prefix of the saved files |
| `--all` | no | compares ReLU / Sigmoid / Tanh with SGD and Adam |

Files produced, for each optimizer/activation combination :

- `saved_model_<optimizer>_<activation>.npy` : the model (weights, biases, **and** standardization parameters) ;
- `saved_model_<optimizer>_<activation>_history.json` : the complete history of the metrics.

### Step 3 : predict

```bash
python predict.py --dataset datasets/data_predict.csv --model saved_model_adam_relu.npy
```

The script displays the binary cross-entropy, the accuracy, the confusion matrix, the precision, the recall and the F1, then writes `predictions.csv` (columns `ID, Prediction, P_M, P_B`).

---

## 5. The basics of a neural network

### 5.1 The neuron

An artificial neuron is a very small function. It receives numbers $x_1, \dots, x_n$, multiplies each of them by a **weight** $w_i$, adds everything together with a **bias** $b$, then applies an **activation function** $f$ :

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}\cdot\mathbf{x} + b$$

$$a = f(z)$$

- the **weights** tell how much each input counts (and in which direction) ;
- the **bias** shifts the triggering threshold ;
- the **activation** introduces non-linearity (section 7).

### 5.2 The layer

A layer groups several neurons that look at the **same inputs**. With $n_{in}$ inputs and $n_{out}$ neurons, all the weights fit into a matrix $W$ of size $n_{out} \times n_{in}$ (one row per neuron) and the biases into a vector $\mathbf{b}$ of size $n_{out}$ :

$$\mathbf{z} = W\mathbf{x} + \mathbf{b} \qquad\qquad \mathbf{a} = f(\mathbf{z})$$

### 5.3 The network

A network stacks layers : the output of one layer is the input of the next.

```
 input           hidden layer 1      hidden layer 2         output
 (30 features)   (24 neurons)        (24 neurons)           (2 neurons)

   x1 ─┐
   x2 ─┤            ○                    ○                    ○  → P(malignant)
   x3 ─┼─────────►  ○  ─────────────►    ○  ──────────────►   ○  → P(benign)
   …   ┤            ⋮                    ⋮
  x30 ─┘            ○                    ○
                  ReLU                  ReLU                softmax
```

By default : `30 → 24 → 24 → 2`. The architecture can be set with `--layer`.

- The **hidden layers** gradually transform the 30 raw measurements into more abstract representations, more useful for deciding.
- The **output layer** has 2 neurons, one per class, with a softmax activation : it produces two probabilities that sum to 1.

### 5.4 Why non-linearity is essential

Without an activation function, chaining two layers amounts to a single one :

$$W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$$

It is still a linear transformation, no matter the number of layers : the network could only draw straight-line decision boundaries. Non-linear activations make it possible to bend these boundaries and to learn complex relationships between the measurements.

### 5.5 Learning = finding the right weights

At the start, the weights are random : the network answers nonsense. **Learning** means adjusting the $W$ and $\mathbf{b}$ of all the layers so that the predictions match the true diagnoses. The `30 → 24 → 24 → 2` network has :

$$(30\times 24 + 24) + (24\times 24 + 24) + (24\times 2 + 2) = 744 + 600 + 50 = 1394 \text{ parameters.}$$

The learning cycle repeats hundreds of times :

1. **Forward** : we compute the predictions with the current weights.
2. **Cost** : we measure how wrong they are.
3. **Backward** : we compute in which direction to change each weight to reduce the error (backpropagation).
4. **Update** : we move each weight slightly in that direction (optimizer).

---

## 6. The forward pass (forward)

Let $\mathbf{a}^{(0)} = \mathbf{x}$ be the input (the 30 standardized features) and $L$ the number of layers. For each layer $l = 1, \dots, L$ :

$$\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \qquad\qquad \mathbf{a}^{(l)} = f_l\big(\mathbf{z}^{(l)}\big)$$

The output of the network is $\hat{\mathbf{y}} = \mathbf{a}^{(L)} = (\hat p_M,\ \hat p_B)$.

### Batch version, as in the code

To process $N$ samples at once, we arrange the inputs in a matrix $A$ of size $N \times n_{in}$ (one sample per row). The code then computes :

$$Z = A_{prev}\, W^\top + \mathbf{b}$$

with the dimensions $(N \times n_{in})\cdot(n_{in}\times n_{out}) = N \times n_{out}$ (`a_prev @ self.W.T + self.b` in `layer.py`). The bias $\mathbf{b}$ is added to each row (*broadcasting*). Using matrices instead of Python loops makes the computation dozens of times faster.

### Small numerical example

One neuron, two inputs, $\mathbf{w} = (0{,}5;\ -1)$, $b = 0{,}2$, input $\mathbf{x} = (2;\ 1)$, ReLU activation :

$$z = 0{,}5\times 2 + (-1)\times 1 + 0{,}2 = 0{,}2 \qquad a = \max(0,\ 0{,}2) = 0{,}2$$

---

## 7. Activation functions

Implemented in `mlp/activations.py`. Each one needs its **derivative**, which will be used for backpropagation.

### ReLU (*Rectified Linear Unit*)

$$\text{relu}(z) = \max(0, z) \qquad\qquad \text{relu}'(z) = \begin{cases}1 & z > 0\\ 0 & z \le 0\end{cases}$$

Very simple and very fast. It does not saturate for positive values (the gradient stays 1), which prevents learning from slowing down in deep networks. Drawback : a neuron whose $z$ stays negative has a zero gradient and no longer improves (*dying ReLU*).

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad\qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

It squashes any number into $]0, 1[$. Proof of the derivative (also in a comment in `activations.py`) :

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \underbrace{\frac{1}{1+e^{-z}}}_{\sigma(z)} \cdot \underbrace{\frac{e^{-z}}{1+e^{-z}}}_{1-\sigma(z)}$$

Drawback : for large $|z|$, the curve is almost flat so $\sigma' \approx 0$. The gradient becomes tiny : this is the **vanishing gradient** (gradient that vanishes), which slows down the learning of deep layers. The code limits $z$ to $[-500, 500]$ before the exponential to avoid a numerical overflow.

### Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad\qquad \tanh'(z) = 1 - \tanh^2(z)$$

Like the sigmoid but in $]-1, 1[$, and **centered on 0**, which often helps the optimization. It also saturates for large $|z|$.

### Softmax (output layer)

The softmax transforms a vector of arbitrary scores (called *logits*) into a **probability distribution** :

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Each output is in $]0,1[$ and their sum is exactly 1. Here, $\hat p_M + \hat p_B = 1$. The predicted class is the one with the highest probability.

**Numerical stability** : $e^{1000}$ overflows. We use the translation invariance of the softmax : subtracting the maximum does not change the result but keeps the exponentials in a safe range :

$$\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - \max(\mathbf z)}}{\sum_j e^{z_j - \max(\mathbf z)}}$$

This is what `softmax()` does in the code (`z - np.max(z, axis=1, keepdims=True)`).

---

## 8. Measuring the error : the cost function

To learn, we need a **number** that says how wrong the network is. The smaller it is, the better : this is the **cost function** (*loss*).

### Label representation (one-hot)

The diagnoses are encoded as follows :

| Diagnosis | Target vector $\mathbf{y}$ |
|---|---|
| Malignant (M) | `[1, 0]` |
| Benign (B) | `[0, 1]` |

### The binary cross-entropy

For one sample, with $p$ the probability of the benign class predicted by the network and $y \in \{0, 1\}$ the true value (1 = benign) :

$$\mathcal{L}(p, y) = -\Big[\,y \ln p + (1-y)\ln(1-p)\,\Big]$$

Over the whole dataset, we take the mean :

$$J = \frac{1}{N}\sum_{k=1}^{N}\mathcal{L}(p_k, y_k)$$

### Why this formula ?

Let's look at the two possible cases :

- If the tumor is benign ($y = 1$) : $\mathcal{L} = -\ln p$. If the network is confidently right ($p \to 1$), $-\ln p \to 0$ : no penalty. If it is confidently wrong ($p \to 0$), $-\ln p \to +\infty$ : huge penalty.
- If the tumor is malignant ($y = 0$) : $\mathcal{L} = -\ln(1-p)$, which works symmetrically.

| Network prediction | Truth | Cost |
|---|---|---|
| 99 % benign | benign | 0,010 |
| 70 % benign | benign | 0,357 |
| 50 % benign | benign | 0,693 |
| 10 % benign | benign | 2,303 |
| 1 % benign | benign | 4,605 |

The cross-entropy doesn't only punish errors, it above all punishes **errors made with confidence**. This is what we want in medicine : a model that says « 99 % sure » and is wrong is much more dangerous than a model that hesitates.

It is also, for classification, a better cost function than the squared error : combined with the softmax, it gives a very simple gradient (section 9) and does not flatten out when the network is very wrong.

### Implementation detail

- `np.clip(y_pred, 1e-15, 1 - 1e-15)` avoids $\ln(0) = -\infty$.
- With two softmax outputs, the code uses column 1 (probability of benign) : since $\hat p_M = 1 - \hat p_B$, this single column is enough.

### Link with the categorical cross-entropy

For two classes, the categorical cross-entropy $-\sum_i y_i \ln \hat p_i$ is **identical** to the binary cross-entropy, because $y_M = 1 - y_B$ and $\hat p_M = 1 - \hat p_B$. That is why the displayed loss (binary) and the gradient used (section 9, written for the softmax) are consistent.

---

## 9. Backpropagation (backpropagation)

This is the heart of learning. The problem : we want to know, for **each of the 1394 parameters**, how the cost $J$ changes if we modify it a little. Mathematically, we want the **gradient** $\dfrac{\partial J}{\partial W^{(l)}}$ and $\dfrac{\partial J}{\partial \mathbf b^{(l)}}$ for each layer.

### 9.1 The intuition

Imagine a mountain whose altitude is the cost $J$ and whose coordinates are all the weights. The gradient indicates **the direction of the steepest upward slope**. To go down (reduce the error), we move in the opposite direction. This is **gradient descent**.

Backpropagation is simply an efficient way of computing this gradient, by applying the **chain rule** from the last layer to the first, reusing the intermediate computations.

### 9.2 The chain rule

If $J$ depends on $a$, which depends on $z$, which depends on $w$ :

$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}$$

The network is a big composition of functions, so the gradient is a product of small pieces, one per layer crossed.

### 9.3 The error of the output layer : softmax + cross-entropy

We define the **error** (*delta*) of a layer as the gradient of the cost with respect to its pre-activations :

$$\boldsymbol\delta^{(l)} = \frac{\partial J}{\partial \mathbf z^{(l)}}$$

For the output layer, with softmax and cross-entropy, there is a remarkable simplification. For one sample, with $\mathcal{L} = -\sum_k y_k \ln p_k$ and $p_k = \text{softmax}(\mathbf z)_k$ :

The derivative of the softmax is :

$$\frac{\partial p_k}{\partial z_i} = p_k(\delta_{ki} - p_i) \qquad (\delta_{ki} = 1 \text{ if } k=i, \text{ otherwise } 0)$$

Therefore :

$$\frac{\partial \mathcal L}{\partial z_i} = -\sum_k \frac{y_k}{p_k}\, p_k(\delta_{ki} - p_i) = -\sum_k y_k(\delta_{ki} - p_i) = -y_i + p_i\underbrace{\sum_k y_k}_{=1}$$

$$\boxed{\boldsymbol\delta^{(L)} = \hat{\mathbf y} - \mathbf y}$$

The gradient at the output is simply **the prediction minus the truth**. It is the first line of `Network.backward` :

```python
delta = y_pred - y_true
```

If the network predicts $(0{,}9;\ 0{,}1)$ for a malignant tumor $(1;\ 0)$, the error is $(-0{,}1;\ +0{,}1)$ : the « malignant » score must be increased and the « benign » score decreased.

### 9.4 The weight gradients of a layer

Once $\boldsymbol\delta^{(l)}$ is known, the gradients of layer $l$ are obtained directly, because $\mathbf z^{(l)} = W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}$ :

$$\frac{\partial J}{\partial W^{(l)}} = \boldsymbol\delta^{(l)}\,\big(\mathbf a^{(l-1)}\big)^\top \qquad\qquad \frac{\partial J}{\partial \mathbf b^{(l)}} = \boldsymbol\delta^{(l)}$$

Interpretation : the gradient of the weight $w_{ij}$ is (the error of neuron $i$) × (the activity of input $j$). A weight is blamed in proportion to the error of the neuron **and** to the importance of the signal that was passing through it.

On a batch of $N$ samples, we **average** (matrices with one sample per row) :

$$dW^{(l)} = \frac{1}{N}\,\Delta^{(l)\top} A^{(l-1)} \qquad\qquad d\mathbf b^{(l)} = \frac{1}{N}\sum_{k=1}^{N}\boldsymbol\delta^{(l)}_k$$

This is exactly `Layer.backward` :

```python
self.dW = (delta.T @ self.a_prev) / N
self.db = np.mean(delta, axis=0)
```

### 9.5 Propagating the error backward

To move to the previous layer, we « send back » the error through the weights, then we modulate it by the derivative of the activation of that layer :

$$\boldsymbol\delta^{(l-1)} = \Big(\big(W^{(l)}\big)^\top\boldsymbol\delta^{(l)}\Big)\ \odot\ f_{l-1}'\big(\mathbf z^{(l-1)}\big)$$

where $\odot$ is the element-wise product.

- $\big(W^{(l)}\big)^\top\boldsymbol\delta^{(l)}$ : each neuron of layer $l-1$ receives the sum of the errors of the neurons it feeds, weighted by the weights of these connections ;
- $f'(\mathbf z)$ : if the activation was saturated (derivative ≈ 0), the error is attenuated ; if the ReLU neuron was off, it is cut off sharply.

In the code (`network.py`), for layer $i$ :

```python
delta = self.layers[i].backward(delta)      # computes dW, db and returns delta @ W
if i > 0:
    delta *= activation_derivative(self.layers[i - 1].z, self.layers[i - 1].activation)
```

(In the « one sample per row » notation, $\boldsymbol\delta^\top W$ is written `delta @ W`.)

### 9.6 Summary of the algorithm

```
FORWARD
    a⁰ = x
    for l = 1 … L :
        zˡ = Wˡ·aˡ⁻¹ + bˡ
        aˡ = fₗ(zˡ)

BACKWARD
    δᴸ = ŷ − y                                   (softmax + cross-entropy)
    for l = L … 1 :
        dWˡ = δˡ · (aˡ⁻¹)ᵀ / N
        dbˡ = mean(δˡ)
        if l > 1 :
            δˡ⁻¹ = (Wˡ)ᵀ·δˡ ⊙ f'ₗ₋₁(zˡ⁻¹)
```

The cost of the computation is of the same order as a forward pass : this is what makes the learning of networks with millions of parameters practicable.

### 9.7 Remark on `softmax_derivative`

The derivative of the softmax is actually a matrix (Jacobian) $\text{diag}(\mathbf s) - \mathbf s \mathbf s^\top$. The code provides it to be complete, but it is **never called** during training : thanks to the simplification of section 9.3, the delta of the output layer is computed directly as $\hat{\mathbf y} - \mathbf y$, and the softmax is only used on the last layer.

---

## 10. The optimizers : SGD and Adam

The gradient gives a direction. The **optimizer** decides **how to move forward** in that direction. Both are in `mlp/optimiseur.py`.

### 10.1 Stochastic gradient descent (SGD)

Update rule, for each parameter $\theta$ (weight or bias) :

$$\theta \leftarrow \theta - \eta\,\frac{\partial J}{\partial \theta}$$

$\eta$ (*eta*) is the **learning rate** : the size of the step.

| Learning rate | Effect |
|---|---|
| too small | very slow learning, risk of staying stuck |
| well chosen | steady descent toward a minimum |
| too large | the step overshoots the hollow : the loss oscillates, or even diverges |

Drawback : a single $\eta$ for all the parameters, and no « memory » of the previous steps.

> We speak of **stochastic** gradient because it is computed on a mini-batch drawn at random and not on all the data : it is a noisy estimate of the true gradient, but much cheaper to compute, and the noise even sometimes helps to get out of bad minima.

### 10.2 Adam (*Adaptive Moment Estimation*)

Adam improves on SGD with two ideas :

1. **Momentum** : keep a moving average of the past gradients (like a ball gaining momentum) to smooth the noise and accelerate in a constant direction.
2. **Adaptive rate** : give each parameter its own step, reduced when its gradients are often large and increased when they are small.

At each step $t$, with $g_t$ the gradient :

**First moment** (mean of the gradients, the « momentum ») :

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t$$

**Second moment** (mean of the squared gradients, the « scale ») :

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$

**Bias correction** (since $m$ and $v$ start at 0, they are underestimated at the beginning) :

$$\hat m_t = \frac{m_t}{1-\beta_1^t} \qquad\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Update** :

$$\theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

Default values used (those of the original paper) : $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\varepsilon = 10^{-8}$ (avoids division by zero), $\eta = 0{,}001$.

The division by $\sqrt{\hat v_t}$ normalizes the step : a parameter with always huge gradients takes moderate steps, a parameter with tiny gradients takes larger steps. Result : Adam is generally **less sensitive to the choice of the learning rate** and converges faster than SGD, which can be verified with `--all`.

> In this project, the optimizer manages its own memory ($m$, $v$, $t$) per layer. The `Layer` class only computes `dW` and `db`.

---

## 11. Weight initialization

If all the weights started at 0, all the neurons of a layer would compute the same thing, would receive the same gradients and would stay identical forever (the **symmetry** problem). We therefore initialize randomly, but not just any way : the size of the weights must keep the signal at a stable amplitude from one layer to the next. Too large, it explodes ; too small, it dies out.

`mlp/layer.py` draws the weights from a uniform distribution $[-\text{limit}, +\text{limit}]$ :

| Activation | Method | Limit |
|---|---|---|
| ReLU | **He** (Kaiming) uniform | $\sqrt{\dfrac{6}{n_{in}}}$ |
| sigmoid, tanh, softmax | **Glorot / Xavier** uniform | $\sqrt{\dfrac{6}{n_{in}+n_{out}}}$ |

The biases start at 0.

- **Xavier** keeps the variance of the activations roughly constant for symmetric activations like tanh.
- **He** doubles the variance compared to Xavier to compensate for the fact that ReLU sets half of the values to zero.

---

## 12. Data standardization

The 30 features don't have the same scale at all : the area (`Area`) is worth several hundreds to several thousands, whereas the smoothness of the contour (`Smoothness`) is worth about 0,1. We center and scale each feature, using the mean $\mu_j$ and the standard deviation $\sigma_j$ **of the training set** :

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

Each feature then has a mean of 0 and a standard deviation of 1.

### Why it is essential

In theory, the weights of a layer could absorb a shift or a change of scale : standardization **does not add capacity** to the network. But it is essential for the **optimization** :

1. **The gradients depend on the scale of the inputs.** The gradient of a weight is proportional to the value of the feature that passes through it (section 9.4). With the same learning rate for all, a feature of the order of 1000 receives gradients thousands of times larger than a feature of the order of 0,1 : the step that suits one makes the other diverge.
2. **The initialization assumes inputs of amplitude ≈ 1.** With an input at 1000, a weight of 0,3 gives a score of 300 and the softmax saturates from the start (probabilities 0 or 1, zero gradients).
3. The landscape of the loss is smoother (fewer narrow and elongated valleys), so the descent is faster and more stable.

### What standardization is not

It does not tell the network « this is the mean value » : it is a simple fixed rescaling that **preserves the differences** (a larger value remains larger).

### Never re-standardize with the new data

At prediction time, we reuse **the mean and the standard deviation of the train**, not those of the new data. If we standardized new data with their own statistics, an « abnormally high » value would become 0 and would be seen as a normal value : the network would be blind to the drift of the data. That is why `mean` and `std` are **saved in the model file** : the model is self-contained, and another training cannot overwrite the normalization parameters by mistake.

---

## 13. The training loop

Implemented in `Network.fit`. Vocabulary :

- **epoch** : one complete pass over the whole training set ;
- **batch** : a small bundle of samples (`--batch_size`, 16 by default) ;
- **iteration** : one update of the weights, on one batch.

With ~455 training samples and batches of 16, one epoch = 29 updates.

```
for each epoch :
    randomly shuffle the samples
    for each mini-batch :
        ŷ = forward(batch)
        backward(ŷ, y_batch)        # computes dW, db
        optimiseur.step()           # updates W and b
    evaluate on the whole train and the whole validation
    record loss, accuracy, precision, recall, F1
    early stopping ?
```

### Why mini-batches ?

| Method | Gradient computed on | Advantages / drawbacks |
|---|---|---|
| Full batch | the whole set | exact gradient, but only 1 update per epoch : slow |
| One sample at a time | 1 sample | frequent updates but very noisy |
| **Mini-batch** | 8 to 64 samples | good compromise : moderate noise, efficient vectorized computation |

### Why shuffle at each epoch ?

If the data is always presented in the same order, the network can learn artifacts of that order, and the batches are always the same. The shuffling (`np.random.permutation`) breaks these correlations.

### Training vs validation

We split the data into two sets :

- **train** : the network learns on it (this is where the weights are modified) ;
- **validation** : the network **never trains on it**. It is used to measure whether it **generalizes** to cases it has never seen.

A large enough network can simply **memorize** the train (**overfitting**, *overfitting*) : excellent score on the train, bad on new cases. Overfitting is seen when `loss` keeps going down while `val_loss` goes back up.

---

## 14. Early stopping

Principle : stop the training when the validation stops improving, rather than continuing until overfitting.

With `--patience N` :

1. at each epoch, we compare `val_loss` to the best value seen so far ;
2. if it improves : we **copy** the current weights (`W.copy()`, `b.copy()`) and reset the wait counter to 0 ;
3. otherwise, we increment the counter ; when it reaches `N`, we stop ;
4. at the end, we **restore the weights of the best epoch**.

The restoration is important : without it, the saved model would be that of the last epoch, whereas the last `N` epochs were degrading. The copies are necessary, otherwise we would keep mere references to arrays that the optimizer keeps modifying.

```bash
python training.py --train datasets/data_train.csv --val datasets/data_predict.csv \
    --epochs 200 --patience 10
```

Typical output :

```
early stopping at epoch 47 (best epoch: 37, val_loss: 0.0612)
> best weights restored (epoch 37)
```

The selected epoch is recorded in the JSON (`best_epoch`), and the legend of the plots displays the values of this epoch.

---

## 15. Evaluation metrics

### The confusion matrix

We choose the **malignant (M) class as the « positive » class**, because it is the one we are trying to detect.

|  | Predicted M | Predicted B |
|---|---|---|
| **True M** | **TP** (true positive) : cancer detected | **FN** (false negative) : cancer missed |
| **True B** | **FP** (false positive) : false alarm | **TN** (true negative) : benign recognized |

### The metrics

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{precision} = \frac{TP}{TP + FP} \qquad\qquad \text{recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2\cdot\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$$

| Metric | Question it answers |
|---|---|
| **Accuracy** | What share of all the answers is correct ? |
| **Precision** | When the model says « malignant », is it right ? (penalizes false alarms) |
| **Recall** (*recall*, sensitivity) | Among the true cancers, how many does it detect ? (penalizes missed cancers) |
| **F1** | A compromise between precision and recall (harmonic mean, pulled down as soon as one of the two is low) |

### Why accuracy is not enough

With 63 % of benign cases, answering « benign » everywhere gives 63 % accuracy but a recall of 0 : no cancer detected. Moreover, the two types of error are not equal :

- a **false negative** (cancer missed) can delay a vital treatment ;
- a **false positive** (false alarm) leads to additional examinations, stressful but rarely serious.

In a medical context, it is therefore the **recall** on the malignant class that is the most telling metric.

### Special cases

If $TP + FP = 0$ (the model never predicts « malignant »), the precision is a 0/0 division : the code returns 0 to avoid the error. Same for the recall and the F1.

---

## 16. Reading the training output

Example of a line displayed at each epoch :

```
epoch 16/70 - loss: 0.0352 - val_loss: 0.0930 - acc: 0.9956 - val_acc: 0.9825 - val_precision: 1.0000 - val_recall: 0.9608 - val_f1: 0.9800
```

| Field | Meaning |
|---|---|
| `epoch 16/70` | 16th complete pass over the data, out of 70 requested |
| `loss` | cross-entropy on the **training** set |
| `val_loss` | cross-entropy on the **validation** set (monitored by early stopping) |
| `acc` / `val_acc` | accuracy on the train / the validation |
| `val_precision` | share of the announced « malignant » that really are |
| `val_recall` | share of the true cancers detected |
| `val_f1` | precision / recall compromise |

All these values are computed on the whole data at the end of the epoch, not on a single batch.

### How to interpret

- `loss` < `val_loss` is normal : the model is better on what it has learned. A modest gap is healthy.
- If `val_loss` **goes up** while `loss` **goes down** : overfitting.
- If `loss` and `val_loss` are both high : underfitting (model too small, unsuitable learning rate, too few epochs).
- The validation set is small (~114 cases) : one case more or less shifts the metrics by 1 to 2 points, so one should not over-interpret the variations from one epoch to the next.

### Learning curves

With `--all`, the program plots for each optimizer (SGD, Adam) the loss and the accuracy of the three activations (ReLU, sigmoid, tanh), as a solid line for the train and dotted for the validation. This makes it possible to visually compare the convergence speeds.

### Recorded history

Each training writes a `*_history.json` containing the optimizer, the activation, the hyperparameters, the selected epoch and all the curves (loss, accuracy, precision, recall, F1, for the train and the validation). This makes it possible to compare several runs without rerunning them.

---

## 17. Glossary

| Term | Definition |
|---|---|
| **Feature** | an input variable (here, one of the 30 measurements) |
| **Label / target** | the correct answer (M or B) |
| **Weights / biases** | parameters learned by the network |
| **Activation** | non-linear function applied to the output of a neuron |
| **Logit** | raw score before the softmax |
| **Forward** | computation of the output from the input |
| **Loss / cost** | number measuring the error of the model |
| **Gradient** | vector of the derivatives of the loss with respect to the parameters |
| **Backpropagation** | efficient computation of the gradient by the chain rule |
| **Learning rate** | size of the update step |
| **Epoch** | one complete pass over the training set |
| **Batch** | subset of samples processed together |
| **Overfitting** | the model memorizes the train instead of generalizing |
| **Generalization** | good behavior on never-seen data |
| **Hyperparameter** | setting fixed before training (learning rate, layer sizes…) |
| **Standardization** | centering at 0 and scaling to a standard deviation of 1 |
| **Early stopping** | stopping the training when the validation no longer improves |

---

## References

- Data : *Wisconsin Diagnostic Breast Cancer (WDBC)*, W. H. Wolberg, W. N. Street, O. L. Mangasarian (UCI Machine Learning Repository).
- Adam : D. P. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*, 2014.
- Glorot initialization : X. Glorot, Y. Bengio, *Understanding the difficulty of training deep feedforward neural networks*, 2010.
- He initialization : K. He et al., *Delving Deep into Rectifiers*, 2015.
- Backpropagation : D. E. Rumelhart, G. E. Hinton, R. J. Williams, *Learning representations by back-propagating errors*, 1986.

### Wikipedia

**Neural networks and machine learning**

- Multilayer perceptron : [https://en.wikipedia.org/wiki/Multilayer_perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- Neural network : [https://en.wikipedia.org/wiki/Neural_network_(machine_learning)](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
- Supervised learning : [https://en.wikipedia.org/wiki/Supervised_learning](https://en.wikipedia.org/wiki/Supervised_learning)
- Binary classification : [https://en.wikipedia.org/wiki/Binary_classification](https://en.wikipedia.org/wiki/Binary_classification)
- Feature : [https://en.wikipedia.org/wiki/Feature_(machine_learning)](https://en.wikipedia.org/wiki/Feature_(machine_learning))
- One-hot : [https://en.wikipedia.org/wiki/One-hot](https://en.wikipedia.org/wiki/One-hot)
- Overfitting : [https://en.wikipedia.org/wiki/Overfitting](https://en.wikipedia.org/wiki/Overfitting)
- Early stopping : [https://en.wikipedia.org/wiki/Early_stopping](https://en.wikipedia.org/wiki/Early_stopping)
- Regularization : [https://en.wikipedia.org/wiki/Regularization_(mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))
- Dropout : [https://en.wikipedia.org/wiki/Dropout_(neural_networks)](https://en.wikipedia.org/wiki/Dropout_(neural_networks))
- Cross-validation : [https://en.wikipedia.org/wiki/Cross-validation_(statistics)](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- Feature scaling : [https://en.wikipedia.org/wiki/Feature_scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- NumPy : [https://en.wikipedia.org/wiki/NumPy](https://en.wikipedia.org/wiki/NumPy)

**Activation functions**

- Activation function : [https://en.wikipedia.org/wiki/Activation_function](https://en.wikipedia.org/wiki/Activation_function)
- ReLU (and dying ReLU) : [https://en.wikipedia.org/wiki/Rectifier_(neural_networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- Sigmoid function : [https://en.wikipedia.org/wiki/Sigmoid_function](https://en.wikipedia.org/wiki/Sigmoid_function)
- Hyperbolic functions (tanh) : [https://en.wikipedia.org/wiki/Hyperbolic_functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
- Softmax : [https://en.wikipedia.org/wiki/Softmax_function](https://en.wikipedia.org/wiki/Softmax_function)
- Logit : [https://en.wikipedia.org/wiki/Logit](https://en.wikipedia.org/wiki/Logit)
- Vanishing gradient problem : [https://en.wikipedia.org/wiki/Vanishing_gradient_problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

**Loss and optimization**

- Loss function : [https://en.wikipedia.org/wiki/Loss_function](https://en.wikipedia.org/wiki/Loss_function)
- Cross-entropy : [https://en.wikipedia.org/wiki/Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy)
- Backpropagation : [https://en.wikipedia.org/wiki/Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- Gradient descent : [https://en.wikipedia.org/wiki/Gradient_descent](https://en.wikipedia.org/wiki/Gradient_descent)
- Stochastic gradient descent (mini-batch, momentum, Adam) : [https://en.wikipedia.org/wiki/Stochastic_gradient_descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- Learning rate : [https://en.wikipedia.org/wiki/Learning_rate](https://en.wikipedia.org/wiki/Learning_rate)
- Moving average : [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)

**Evaluation metrics**

- Confusion matrix : [https://en.wikipedia.org/wiki/Confusion_matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- Accuracy and precision : [https://en.wikipedia.org/wiki/Accuracy_and_precision](https://en.wikipedia.org/wiki/Accuracy_and_precision)
- Precision and recall : [https://en.wikipedia.org/wiki/Precision_and_recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- Sensitivity and specificity : [https://en.wikipedia.org/wiki/Sensitivity_and_specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- F-score : [https://en.wikipedia.org/wiki/F-score](https://en.wikipedia.org/wiki/F-score)
- Harmonic mean : [https://en.wikipedia.org/wiki/Harmonic_mean](https://en.wikipedia.org/wiki/Harmonic_mean)

**Mathematics**

- Derivative : [https://en.wikipedia.org/wiki/Derivative](https://en.wikipedia.org/wiki/Derivative)
- Partial derivative : [https://en.wikipedia.org/wiki/Partial_derivative](https://en.wikipedia.org/wiki/Partial_derivative)
- Chain rule : [https://en.wikipedia.org/wiki/Chain_rule](https://en.wikipedia.org/wiki/Chain_rule)
- Gradient : [https://en.wikipedia.org/wiki/Gradient](https://en.wikipedia.org/wiki/Gradient)
- Jacobian matrix and determinant : [https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
- Kronecker delta : [https://en.wikipedia.org/wiki/Kronecker_delta](https://en.wikipedia.org/wiki/Kronecker_delta)
- Linear map : [https://en.wikipedia.org/wiki/Linear_map](https://en.wikipedia.org/wiki/Linear_map)
- Matrix : [https://en.wikipedia.org/wiki/Matrix_(mathematics)](https://en.wikipedia.org/wiki/Matrix_(mathematics))
- Matrix multiplication : [https://en.wikipedia.org/wiki/Matrix_multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)
- Transpose : [https://en.wikipedia.org/wiki/Transpose](https://en.wikipedia.org/wiki/Transpose)
- Hadamard product : [https://en.wikipedia.org/wiki/Hadamard_product_(matrices)](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
- Exponential function : [https://en.wikipedia.org/wiki/Exponential_function](https://en.wikipedia.org/wiki/Exponential_function)
- Natural logarithm : [https://en.wikipedia.org/wiki/Natural_logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
- Uniform distribution : [https://en.wikipedia.org/wiki/Continuous_uniform_distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
- Variance : [https://en.wikipedia.org/wiki/Variance](https://en.wikipedia.org/wiki/Variance)
- Standard deviation : [https://en.wikipedia.org/wiki/Standard_deviation](https://en.wikipedia.org/wiki/Standard_deviation)
- Standard score (standardization) : [https://en.wikipedia.org/wiki/Standard_score](https://en.wikipedia.org/wiki/Standard_score)

**Numerical computing**

- Numerical stability : [https://en.wikipedia.org/wiki/Numerical_stability](https://en.wikipedia.org/wiki/Numerical_stability)
- Arithmetic overflow : [https://en.wikipedia.org/wiki/Arithmetic_overflow](https://en.wikipedia.org/wiki/Arithmetic_overflow)