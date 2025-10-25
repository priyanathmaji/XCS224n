# Stanford CS224N Assignment 2: Word2Vec and Negative Sampling - Complete Algorithm Documentation

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Algorithm Components](#algorithm-components)
4. [Implementation Details](#implementation-details)
5. [Numpy Tricks and Optimizations](#numpy-tricks-and-optimizations)
6. [Complete Algorithm Flow](#complete-algorithm-flow)
7. [Key Formulas Reference](#key-formulas-reference)

## Overview

This assignment implements the Word2Vec Skip-gram model with two different approaches for computing the loss and gradients:
1. **Naive Softmax** - Uses full softmax over entire vocabulary
2. **Negative Sampling** - Approximates softmax using sampled negative examples

### Core Concept
Word2Vec learns distributed representations of words by predicting context words given a center word (Skip-gram) or predicting center word given context (CBOW). This implementation focuses on Skip-gram.

## Mathematical Foundations

### Skip-gram Objective
For a center word `c` and context words `o`, the objective is to maximize:
```
J = log P(o|c) = log P(w_{o} | w_c)
```

### Two Approaches to Compute P(o|c)

#### 1. Naive Softmax
```
P(o|c) = exp(u_o^T v_c) / Σ_{w=1}^{V} exp(u_w^T v_c)
```
Where:
- `v_c` = center word vector (embedding dimension d)
- `u_o` = outside word vector for word o
- `V` = vocabulary size

#### 2. Negative Sampling
Instead of computing over entire vocabulary, sample K negative examples:
```
J = log σ(u_o^T v_c) + Σ_{k=1}^{K} log σ(-u_k^T v_c)
```
Where:
- `σ(x) = 1/(1 + exp(-x))` is sigmoid function
- `u_k` are vectors of negative samples

## Algorithm Components

### 1. Sigmoid Function
```python
def sigmoid(x):
    s = 1/ (1 + np.exp(-x))
    return s
```
**Mathematical Formula:** `σ(x) = 1/(1 + e^(-x))`

**Purpose:** Used in negative sampling for binary classification of positive vs negative examples.

### 2. Softmax Function (Numerically Stable)
```python
def softmax(x):
    if len(x.shape) > 1:
        # Matrix case
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))  # Subtract max for stability
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector case
        tmp = np.max(x)
        x -= tmp  # Numerical stability trick
        x = np.exp(x)
        x /= np.sum(x)
    return x
```

**Key Trick:** Subtracts maximum value before exponentiation to prevent overflow.
**Mathematical Formula:** `softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))`

### 3. Naive Softmax Loss and Gradient

```python
def naive_softmax_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset):
    # Compute softmax probabilities
    y_hat = softmax(np.dot(outside_vectors, center_word_vec))  
    
    # Loss: negative log probability of true outside word
    loss = -np.log(y_hat[outside_word_idx])
    
    # Gradient computation
    y_hat[outside_word_idx] = y_hat[outside_word_idx] - 1  # y_hat - y (one-hot)
    grad_center_vec = np.dot(y_hat, outside_vectors)       # ∂J/∂v_c
    grad_outside_vecs = np.outer(y_hat, center_word_vec)   # ∂J/∂U
    
    return loss, grad_center_vec, grad_outside_vecs
```

**Key Mathematical Insights:**
- **Loss:** `J = -log(P(o|c)) = -log(y_hat[o])`
- **Gradient w.r.t center vector:** `∂J/∂v_c = U^T(ŷ - y)`
- **Gradient w.r.t outside vectors:** `∂J/∂U = (ŷ - y) ⊗ v_c` (outer product)

**Critical Numpy Trick:** 
```python
y_hat[outside_word_idx] -= 1  # Converts softmax output to gradient form
```
This implements `ŷ - y` where `y` is one-hot vector with 1 at true outside word index.

### 4. Negative Sampling Loss and Gradient

```python
def neg_sampling_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset, K=10):
    # Get K negative samples + 1 positive sample
    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices
    
    # Initialize gradients
    grad_center_vec = np.zeros(center_word_vec.shape)
    grad_outside_vecs = np.zeros(outside_vectors.shape)
    
    # Labels: +1 for positive sample, -1 for negative samples
    labels = np.array([1] + [-1 for k in range(K)])
    vecs = outside_vectors[indices, :]  # Shape: (K+1, embedding_dim)
    
    # Compute sigmoid of labeled dot products
    t = sigmoid(vecs.dot(center_word_vec) * labels)  # Shape: (K+1,)
    loss = -np.sum(np.log(t))
    
    # Gradient computation
    delta = labels * (t - 1)  # ∂J/∂(u_i^T v_c) for each sample
    
    # Gradient w.r.t center vector
    grad_center_vec = delta.reshape((1, K + 1)).dot(vecs).flatten()
    
    # Gradient w.r.t outside vectors  
    grad_outside_vecs_temp = delta.reshape((K + 1, 1)).dot(
        center_word_vec.reshape((1, center_word_vec.shape[0]))
    )
    
    # Accumulate gradients only for sampled indices
    for k in range(K + 1):
        grad_outside_vecs[indices[k]] += grad_outside_vecs_temp[k, :]
    
    return loss, grad_center_vec, grad_outside_vecs
```

**Mathematical Formulas:**
- **Loss:** `J = -Σ log σ(l_i * u_i^T v_c)` where `l_i ∈ {+1, -1}`
- **Gradient w.r.t center:** `∂J/∂v_c = Σ δ_i * u_i` where `δ_i = l_i(σ(l_i * u_i^T v_c) - 1)`
- **Gradient w.r.t outside:** `∂J/∂u_i = δ_i * v_c`

**Key Numpy Tricks:**
1. **Vectorized sigmoid:** `sigmoid(vecs.dot(center_word_vec) * labels)` computes all K+1 sigmoids at once
2. **Efficient outer product:** `delta.reshape((K + 1, 1)).dot(center_word_vec.reshape((1, d)))` creates (K+1, d) matrix
3. **Sparse gradient updates:** Only update `grad_outside_vecs[indices[k]]` for sampled words

### 5. Skip-gram Model Implementation

```python
def skipgram(current_center_word, outside_words, word2ind, center_word_vectors, 
             outside_vectors, dataset, word2vec_loss_and_gradient):
    
    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)  # (V, d)
    grad_outside_vectors = np.zeros(outside_vectors.shape)  # (V, d)
    
    center_word_idx = word2ind[current_center_word]
    center_word_vec = center_word_vectors[center_word_idx]
    
    # For each context word, compute loss and gradients
    for outside_word in outside_words:
        outside_word_idx = word2ind[outside_word]
        c_loss, c_grad_center, c_grad_outside = word2vec_loss_and_gradient(
            center_word_vec, outside_word_idx, outside_vectors, dataset
        )
        
        loss += c_loss
        grad_center_vecs[center_word_idx] += c_grad_center  # Only update center word
        grad_outside_vectors += c_grad_outside              # Update all outside vectors
    
    return loss, grad_center_vecs, grad_outside_vectors
```

**Key Insights:**
- **Center gradient:** Only the row corresponding to current center word is updated
- **Outside gradients:** All vocabulary vectors can be updated (sparse in negative sampling)
- **Loss accumulation:** Sum losses from all context words

### 6. SGD Wrapper and Training Loop

```python
def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset, window_size, 
                        word2vec_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)  # (2V, d) - stores both center and outside vectors
    
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]   # First half: center vectors
    outside_vectors = word_vectors[int(N / 2):, :]       # Second half: outside vectors
    
    for i in range(batchsize):
        window_size_1 = random.randint(1, window_size)
        center_word, context = dataset.get_random_context(window_size_1)
        
        c, gin, gout = word2vec_model(center_word, context, word2ind, 
                                     center_word_vectors, outside_vectors, 
                                     dataset, word2vec_loss_and_gradient)
        
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize      # Update center vector gradients
        grad[int(N / 2):, :] += gout / batchsize     # Update outside vector gradients
    
    return loss, grad
```

**Key Design Decisions:**
- **Dual representation:** Each word has both center and outside vectors
- **Parameter packing:** `word_vectors` contains both center and outside vectors stacked
- **Batch processing:** Averages gradients over multiple random contexts

### 7. Stochastic Gradient Descent

```python
def sgd(f, x0, step, iterations, postprocessing=None, use_saved=False, PRINT_EVERY=10):
    ANNEAL_EVERY = 20000
    
    x = x0
    exploss = None
    
    for iter in range(start_iter + 1, iterations + 1):
        loss, grad = f(x)          # Compute loss and gradients
        x -= step * grad           # SGD update step
        x = postprocessing(x)      # Normalize word vectors
        
        # Learning rate annealing
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
            
        # Exponential moving average of loss for smoother display
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))
    
    return x
```

**Optimization Tricks:**
1. **Learning rate annealing:** Reduces learning rate by half every 20,000 iterations
2. **Exponential moving average:** Smooths loss display for better monitoring
3. **Postprocessing:** Normalizes word vectors to unit length after each update

## Numpy Tricks and Optimizations

### 1. Broadcasting and Vectorization
```python
# Instead of loops, use broadcasting
y_hat = softmax(np.dot(outside_vectors, center_word_vec))  # (V,) = (V,d) @ (d,)

# Outer product for gradient computation
grad_outside_vecs = np.outer(y_hat, center_word_vec)  # (V,d) = (V,) ⊗ (d,)
```

### 2. Numerical Stability
```python
# Subtract max before exp() to prevent overflow
tmp = np.max(x, axis=1)
x -= tmp.reshape((x.shape[0], 1))
x = np.exp(x)
```

### 3. Efficient Reshaping
```python
# Reshape for matrix multiplication
delta.reshape((1, K + 1)).dot(vecs).flatten()  # (1,K+1) @ (K+1,d) -> (1,d) -> (d,)
```

### 4. In-place Operations
```python
y_hat[outside_word_idx] -= 1  # Modify specific element in-place
grad_outside_vecs[indices[k]] += grad_outside_vecs_temp[k, :]  # Sparse update
```

### 5. Advanced Indexing
```python
vecs = outside_vectors[indices, :]  # Select multiple rows efficiently
labels = np.array([1] + [-1 for k in range(K)])  # Vectorized label creation
```

## Complete Algorithm Flow

### Training Process
1. **Initialize:** Random word vectors for center and outside representations
2. **For each training iteration:**
   - Sample random context window from corpus
   - Extract center word and context words
   - **For each context word:**
     - Compute loss and gradients using chosen method (naive softmax or negative sampling)
     - Accumulate gradients for center and outside vectors
   - Update parameters using SGD
   - Normalize word vectors to unit length

### Gradient Computation Pipeline
```
Input: center_word, outside_words
    ↓
For each outside_word:
    ↓ 
Compute: loss, grad_center, grad_outside
    ↓
Accumulate: total_loss, center_gradients[center_idx], outside_gradients
    ↓
Output: total_loss, accumulated_gradients
```

## Key Formulas Reference

### Loss Functions
1. **Naive Softmax Loss:**
   ```
   J = -log(exp(u_o^T v_c) / Σ_{w=1}^{V} exp(u_w^T v_c))
   ```

2. **Negative Sampling Loss:**
   ```
   J = -log σ(u_o^T v_c) - Σ_{k=1}^{K} log σ(-u_k^T v_c)
   ```

### Gradient Formulas
1. **Naive Softmax Gradients:**
   ```
   ∂J/∂v_c = U^T(ŷ - y)
   ∂J/∂u_w = (ŷ_w - y_w) * v_c
   ```

2. **Negative Sampling Gradients:**
   ```
   ∂J/∂v_c = Σ_{i ∈ {o,k_1,...,k_K}} l_i(σ(l_i u_i^T v_c) - 1) * u_i
   ∂J/∂u_i = l_i(σ(l_i u_i^T v_c) - 1) * v_c
   ```

### Utility Functions
1. **Sigmoid:** `σ(x) = 1/(1 + e^(-x))`
2. **Softmax:** `softmax(x_i) = exp(x_i) / Σ exp(x_j)`
3. **Stable Softmax:** `softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))`

## Implementation Insights

### Why Two Vector Representations?
- **Center vectors (V):** Used when word appears as center/target
- **Outside vectors (U):** Used when word appears in context
- **Benefit:** Increases model capacity and training stability

### Negative Sampling Benefits
- **Computational efficiency:** O(K) instead of O(V) per prediction
- **Better gradients:** Avoids vanishing gradients from small softmax probabilities
- **Scalability:** Essential for large vocabularies

### Gradient Accumulation Strategy
- **Center gradients:** Only update current center word's vector
- **Outside gradients:** Update all context words' vectors (or sampled subset)
- **Memory efficiency:** Sparse updates reduce computational overhead

This implementation showcases efficient numpy usage, mathematical rigor, and practical optimizations essential for training word embeddings on large corpora.