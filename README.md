
[![GitHub release](https://img.shields.io/github/release/stephenhky/npdict.svg?maxAge=3600)](https://github.com/stephenhky/npdict/releases)
[![pypi](https://img.shields.io/pypi/v/npdict.svg?maxAge=3600)](https://pypi.org/project/npdict/)
[![download](https://img.shields.io/pypi/dm/npdict.svg?maxAge=2592000&label=installs&color=%2327B1FF)](https://pypi.org/project/npdict/)
[![Documentation Status](https://readthedocs.org/projects/npdict/badge/?version=latest)](https://npdict.readthedocs.io/en/latest/?badge=latest)

# `npdict`: Python Package for Dictionary Wrappers for Numpy Arrays

This Python package, `npdict`, aims at facilitating holding numerical
values in a Python dictionary, but at the same time retaining the
ultra-high performance supported by NumPy. It supports an object
which is a Python dictionary on the surface, but numpy behind the
back, facilitating fast assignment and retrieval of values
and fast computation of numpy arrays.

## Installation

To install, in your terminal, simply enter:

```
pip install npdict
```

## Quickstart

### Instantiation

Suppose you are doing a similarity dictionary between two sets of words.
And each of these sets have words:

```
document1 = ['president', 'computer', 'tree']
document2 = ['chairman', 'abacus', 'trees']
```

And you can build a dictionary like this:

```
import numpy as np
from npdict import NumpyNDArrayWrappedDict

similarity_dict = NumpyNDArrayWrappedDict([document1, document2])
```

An `npdict.NumpyNDArrayWrappedDict` instance is instantiated. It is 
a Python dict:

```
isinstance(similarity_dict, dict)  # which gives `True`
```

It has a matrix inside with default value 0.0 (and the initial default value can
be changed to other values when the instance is instantiated.)

```
similarity_dict.to_numpy()
```
giving
```
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
```

### Value Assignments

Now you can assign values just like what you do to a Python dictionary:

```
similarity_dict['president', 'chairman'] = 0.9
similarity_dict['computer', 'abacus'] = 0.7
similarity_dict['tree', 'trees'] = 0.95
```

And it has changed the inside numpy array to be:

```
array([[0.9  , 0.  , 0.  ],
       [0.  , 0.7 , 0.  ],
       [0.  , 0.  , 0.95]])
```

### Generation of New Object from the Old One

If you want to create another dict using the same words, but 
a manipulation of the original value, 25 percent discount
of the original one for example, you can do something like this:

```
new_similarity_dict = similarity_dict.generate_dict(similarity_dict.to_numpy()*0.75)
```

And you got a new dictionary with numpy array to be:

```
new_similarity_dict.to_numpy()
```
giving
```
array([[0.675 , 0.    , 0.    ],
       [0.    , 0.525 , 0.    ],
       [0.    , 0.    , 0.7125]])
```

This is a simple operation. But the design of this wrapped Python
dictionary is that you can perform any fast or optimized operation
on your numpy array (using numba or Cython, for examples),
while retaining the keywords as your dictionary.

### Retrieval of Values

At the same time, you can set new values just like above, or retrieve
values as if it is a Python dictionary:

```
similarity_dict['president', 'chairman']
```

### Conversion to a Python Dictionary

You can also convert this to an ordinary Python dictionary:

```
raw_similarity_dict = similarity_dict.to_dict()
```

### Instantiation from a Python Dictionary

And you can convert a Python dictionary of this type back to 
`npdict.NumpyNDArrayWrappedDict` by (recommended)

```
new_similarity_dict_2 = NumpyNDArrayWrappedDict.from_dict_given_keywords([document1, document2], raw_similarity_dict)
```

Or you can even do this (not recommended):

```
new_similarity_dict_3 = NumpyNDArrayWrappedDict.from_dict(raw_similarity_dict)
```

It is not recommended because the order of the keys are not retained in this way.
Use it with caution.

## Working with Sparse Arrays

For large, sparse matrices where most elements are zero, using `SparseArrayWrappedDict` can be more memory-efficient than the standard `NumpyNDArrayWrappedDict`.

### Instantiation

Similar to the regular dictionary wrapper, you can instantiate a sparse array wrapper:

```python
from npdict import SparseArrayWrappedDict

document1 = ['president', 'computer', 'tree', 'car', 'house', 'book']
document2 = ['chairman', 'abacus', 'trees', 'vehicle', 'building', 'paper']

# Create a sparse dictionary - efficient for large, sparse matrices
sparse_similarity_dict = SparseArrayWrappedDict([document1, document2])
```

### Value Assignments

Assign values just like with a regular dictionary:

```python
# Only assign values for the few non-zero elements
sparse_similarity_dict['president', 'chairman'] = 0.9
sparse_similarity_dict['computer', 'abacus'] = 0.7
sparse_similarity_dict['tree', 'trees'] = 0.95
```

The sparse implementation only stores the non-zero values, making it memory-efficient for large, sparse matrices.

### Converting Between Formats

You can convert between dense and sparse formats:

```python
# Convert to NumPy array (dense format)
dense_array = sparse_similarity_dict.to_numpy()

# Convert to COO format (another sparse format)
coo_array = sparse_similarity_dict.to_coo()

# Get the underlying DOK (Dictionary of Keys) sparse array
dok_array = sparse_similarity_dict.to_dok()
```

### Generating New Dictionaries

You can generate new dictionaries from existing ones, with options to convert between sparse and dense formats:

```python
# Generate a new sparse dictionary
new_sparse_dict = sparse_similarity_dict.generate_dict(
    sparse_similarity_dict.to_coo() * 0.75
)

# Generate a dense dictionary from a sparse one
dense_dict = sparse_similarity_dict.generate_dict(
    sparse_similarity_dict.to_numpy(),
    dense=True  # This parameter converts to a dense NumpyNDArrayWrappedDict
)
```

### When to Use Sparse Arrays

Use `SparseArrayWrappedDict` when:
- Your data is mostly zeros (sparse)
- You're working with large dimensions where memory usage is a concern
- You need to perform operations that are optimized for sparse matrices

Use `NumpyNDArrayWrappedDict` when:
- Your data has few zeros (dense)
- You need faster element-wise access
- You're working with smaller dimensions where memory usage is less of a concern