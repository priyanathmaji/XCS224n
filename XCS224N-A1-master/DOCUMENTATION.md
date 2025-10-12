# Stanford NLP Assignment - Word Embeddings Documentation

## Project Overview

This project implements word embeddings using co-occurrence matrices and dimensionality reduction techniques. It processes a Reuters corpus to create word representations and visualizes them in 2D space using SVD (Singular Value Decomposition).

## Table of Contents

- [Project Structure](#project-structure)
- [Core Functions](#core-functions)
- [Algorithm Workflow](#algorithm-workflow)
- [Implementation Details](#implementation-details)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Example Output](#example-output)
- [Performance Considerations](#performance-considerations)

## Project Structure

```
XCS224N-A1-master/
├── src/
│   ├── submission.py      # Main implementation file
│   ├── utils.py          # Utility functions for corpus reading
│   ├── grader.py         # Autograder functionality
│   └── graderUtil.py     # Grader utilities
├── tex/                  # LaTeX files for assignment
├── A1.pdf               # Assignment instructions
├── submission.pdf       # Final submission
└── README.md            # This documentation
```

## Core Functions

### 1. `distinct_words(corpus)`

**Purpose**: Extracts unique words from the entire corpus and creates a vocabulary.

**Parameters**:
- `corpus` (list of list of strings): Collection of documents, each containing words

**Returns**:
- `corpus_words` (list of strings): Sorted list of unique words
- `num_corpus_words` (integer): Total count of unique words

**Algorithm**:
```python
def distinct_words(corpus):
    corpus_words = []
    num_corpus_words = 0
    
    for document in corpus:
        for word in document:
            if word not in corpus_words:
                corpus_words.append(word)
                num_corpus_words = num_corpus_words + 1
    
    corpus_words = sorted(corpus_words)
    return corpus_words, num_corpus_words
```

**Time Complexity**: O(V × D × W) where V=vocabulary size, D=documents, W=words per document

### 2. `compute_co_occurrence_matrix(corpus, window_size=4)`

**Purpose**: Creates a matrix showing how frequently words appear together within a context window.

**Parameters**:
- `corpus` (list of list of strings): Corpus of documents
- `window_size` (int): Size of context window (default=4)

**Returns**:
- `M` (numpy matrix): Co-occurrence matrix of shape (vocab_size × vocab_size)
- `word2Ind` (dict): Maps words to matrix indices

**Algorithm**:
```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    words, num_words = distinct_words(corpus)
    word2Ind = {words[i]:i for i in range(num_words)}
    M = np.zeros((num_words, num_words))

    for document in corpus:
        for idx, word in enumerate(document):
            center_Index = word2Ind[word]
            
            # Words before center word
            for context_index in range(max(0, idx - window_size), idx):
                context_word = document[context_index]
                context_index = word2Ind[context_word]
                M[center_Index, context_index] += 1
            
            # Words after center word
            for context_index in range(idx + 1, min(idx + window_size + 1, len(document))):
                context_word = document[context_index]
                context_index = word2Ind[context_word]
                M[center_Index, context_index] += 1
    
    return M, word2Ind
```

**Window Logic**:
- **Before center**: `range(max(0, idx-window_size), idx)`
- **After center**: `range(idx+1, min(idx+window_size+1, len(document)))`
- **Edge handling**: Automatically adjusts for document boundaries

### 3. `reduce_to_k_dim(M, k=2)`

**Purpose**: Reduces high-dimensional co-occurrence matrix to k dimensions using Truncated SVD.

**Parameters**:
- `M` (numpy matrix): Co-occurrence matrix of shape (vocab_size × vocab_size)
- `k` (int): Target embedding dimensions (default=2)

**Returns**:
- `M_reduced` (numpy matrix): Reduced embeddings of shape (vocab_size × k)

**Algorithm**:
```python
def reduce_to_k_dim(M, k=2):
    np.random.seed(4355)
    n_iter = 10
    
    truncatedSVD = TruncatedSVD(n_components=k, n_iter=n_iter)
    M_reduced = truncatedSVD.fit_transform(M)
    
    return M_reduced
```

**Mathematical Foundation**:
- Uses **Truncated SVD**: M ≈ U × Σ × V^T
- Returns: U × Σ (left singular vectors scaled by singular values)
- Captures most important semantic relationships in reduced space

### 4. `main()` - Complete Workflow

**Purpose**: Orchestrates the entire pipeline from corpus processing to visualization.

**Process**:
1. Read Reuters corpus
2. Generate co-occurrence matrix
3. Reduce dimensionality to 2D
4. Normalize embeddings to unit length
5. Create visualization

## Algorithm Workflow

```
Reuters Corpus
    ↓
distinct_words() → [word_list, count]
    ↓
compute_co_occurrence_matrix() → [M_matrix, word2Ind]
    ↓
reduce_to_k_dim() → [M_reduced]
    ↓
Normalization → [M_normalized]
    ↓
Visualization → PNG file
```

## Implementation Details

### Co-occurrence Matrix Properties

1. **Symmetry**: The matrix is **not necessarily symmetric** because:
   - M[i,j] counts when word i is center and word j is in context
   - M[j,i] counts when word j is center and word i is in context
   - These can be different based on word positions

2. **Sparsity**: Most entries are zero since most word pairs don't co-occur

3. **Context Window**: 
   ```
   Example: "The quick brown fox jumps" (window_size=2)
   For "brown" (center):
   - Context: ["The", "quick", "fox", "jumps"]
   ```

### Edge Case Handling

- **Document boundaries**: Words near start/end have smaller context windows
- **Single word documents**: Handled gracefully with empty context
- **Duplicate words**: Each occurrence contributes to co-occurrence counts

### Normalization

```python
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]
```

- Converts each word vector to unit length
- Enables cosine similarity comparisons
- Uses numpy broadcasting for efficiency

## Dependencies

```python
import numpy as np              # Matrix operations
import matplotlib.pyplot as plt # Visualization
from sklearn.decomposition import TruncatedSVD  # Dimensionality reduction
from utils import read_corpus   # Custom corpus reader
```

**Installation**:
```bash
pip install numpy matplotlib scikit-learn nltk
```

**NLTK Data**:
```python
import nltk
nltk.download('reuters')
```

## Usage

### Basic Usage

```python
from submission import *

# Read corpus
reuters_corpus = read_corpus()

# Generate embeddings
M_co_occurrence, word2Ind = compute_co_occurrence_matrix(reuters_corpus)
M_reduced = reduce_to_k_dim(M_co_occurrence, k=2)

# Normalize
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_normalized = M_reduced / M_lengths[:, np.newaxis]
```

### Customization

```python
# Different window size
M_co_occurrence, word2Ind = compute_co_occurrence_matrix(reuters_corpus, window_size=6)

# Higher dimensions
M_reduced = reduce_to_k_dim(M_co_occurrence, k=50)

# Different category
reuters_corpus = read_corpus(category="trade")
```

### Running the Complete Pipeline

```bash
cd src
python submission.py
```

This will:
1. Process the Reuters corpus
2. Generate co-occurrence matrix
3. Create 2D embeddings
4. Save visualization as `co_occurrence_embeddings_(soln).png`

## Example Output

### Sample Vocabulary
```
['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
```

### Co-occurrence Matrix Structure
```
       oil  energy  industry  petroleum
oil    0.0    15.2      8.7       12.1
energy 15.2   0.0      22.3        6.8
...
```

### 2D Embeddings
Words with similar meanings cluster together in the reduced space:
- **Energy-related**: ['oil', 'petroleum', 'energy']
- **Geographic**: ['ecuador', 'venezuela', 'kuwait']
- **Quantitative**: ['barrels', 'bpd', 'output']

## Performance Considerations

### Time Complexity
- **distinct_words()**: O(D × W × V) where D=documents, W=words/doc, V=vocab size
- **compute_co_occurrence_matrix()**: O(D × W × window_size)
- **reduce_to_k_dim()**: O(V² × k × iterations)

### Space Complexity
- **Co-occurrence matrix**: O(V²) - can be large for big vocabularies
- **Reduced embeddings**: O(V × k) - much smaller

### Optimization Opportunities

1. **Use sparse matrices** for large vocabularies:
   ```python
   from scipy.sparse import csr_matrix
   M = csr_matrix((num_words, num_words))
   ```

2. **Vectorized operations** instead of loops:
   ```python
   # Use set operations for distinct_words
   corpus_words = set()
   for document in corpus:
       corpus_words.update(document)
   ```

3. **Memory management** for large corpora:
   - Process in batches
   - Use memory mapping for large matrices

## Mathematical Foundation

### Co-occurrence Intuition
Words that appear in similar contexts tend to have similar meanings (distributional hypothesis).

### SVD Decomposition
- **Purpose**: Captures latent semantic relationships
- **Effect**: Similar words map to nearby points in reduced space
- **Benefits**: Noise reduction, computational efficiency

### Cosine Similarity
After normalization, dot product gives cosine similarity:
```python
similarity = np.dot(word_vector_1, word_vector_2)
```

## Testing and Validation

### Running Tests
```bash
python grader.py
```

### Manual Validation
1. Check distinct words are sorted and unique
2. Verify co-occurrence matrix is non-negative
3. Ensure reduced embeddings have correct dimensions
4. Validate visualization shows semantic clustering

## Common Issues and Solutions

### Issue 1: Memory Error
**Problem**: Large vocabulary causes memory issues
**Solution**: Use sparse matrices or reduce vocabulary size

### Issue 2: Slow Performance
**Problem**: Nested loops are inefficient
**Solution**: Vectorize operations where possible

### Issue 3: Empty Embeddings
**Problem**: SVD fails on zero matrix
**Solution**: Check corpus preprocessing and co-occurrence logic

## File Descriptions

- **`submission.py`**: Main implementation with core functions
- **`utils.py`**: Utility functions for reading Reuters corpus
- **`grader.py`**: Automated testing and grading
- **`environment.yml`**: Conda environment specification

## Conclusion

This implementation provides a foundational understanding of:
- Statistical word embeddings
- Co-occurrence-based semantic modeling
- Dimensionality reduction techniques
- Text preprocessing and analysis

The resulting embeddings capture semantic relationships through statistical patterns in the Reuters corpus, forming the basis for more advanced NLP applications like Word2Vec and GloVe.

## References

- [Reuters Corpus](https://www.nltk.org/book/ch02.html#reuters-corpus)
- [Truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)
- [Stanford CS224N Course](https://web.stanford.edu/class/cs224n/)
