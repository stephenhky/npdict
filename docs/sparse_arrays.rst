Working with Sparse Arrays
=========================

For large, sparse matrices where most elements are zero, using ``SparseArrayWrappedDict`` can be more memory-efficient than the standard ``NumpyNDArrayWrappedDict``.

Instantiation
------------

Similar to the regular dictionary wrapper, you can instantiate a sparse array wrapper:

.. code-block:: python

    from npdict import SparseArrayWrappedDict

    document1 = ['president', 'computer', 'tree', 'car', 'house', 'book']
    document2 = ['chairman', 'abacus', 'trees', 'vehicle', 'building', 'paper']

    # Create a sparse dictionary - efficient for large, sparse matrices
    sparse_similarity_dict = SparseArrayWrappedDict([document1, document2])

Value Assignments
---------------

Assign values just like with a regular dictionary:

.. code-block:: python

    # Only assign values for the few non-zero elements
    sparse_similarity_dict['president', 'chairman'] = 0.9
    sparse_similarity_dict['computer', 'abacus'] = 0.7
    sparse_similarity_dict['tree', 'trees'] = 0.95

The sparse implementation only stores the non-zero values, making it memory-efficient for large, sparse matrices.

Converting Between Formats
------------------------

You can convert between dense and sparse formats:

.. code-block:: python

    # Convert to NumPy array (dense format)
    dense_array = sparse_similarity_dict.to_numpy()

    # Convert to COO format (another sparse format)
    coo_array = sparse_similarity_dict.to_coo()

    # Get the underlying DOK (Dictionary of Keys) sparse array
    dok_array = sparse_similarity_dict.to_dok()

Generating New Dictionaries
-------------------------

You can generate new dictionaries from existing ones, with options to convert between sparse and dense formats:

.. code-block:: python

    # Generate a new sparse dictionary
    new_sparse_dict = sparse_similarity_dict.generate_dict(
        sparse_similarity_dict.to_coo() * 0.75
    )

    # Generate a dense dictionary from a sparse one
    dense_dict = sparse_similarity_dict.generate_dict(
        sparse_similarity_dict.to_numpy(),
        dense=True  # This parameter converts to a dense NumpyNDArrayWrappedDict
    )

When to Use Sparse Arrays
-----------------------

Use ``SparseArrayWrappedDict`` when:

* Your data is mostly zeros (sparse)
* You're working with large dimensions where memory usage is a concern
* You need to perform operations that are optimized for sparse matrices

Use ``NumpyNDArrayWrappedDict`` when:

* Your data has few zeros (dense)
* You need faster element-wise access
* You're working with smaller dimensions where memory usage is less of a concern

Memory Usage Comparison
---------------------

For a simple comparison, consider a 1000x1000 matrix with only 1% non-zero elements:

.. code-block:: python

    import numpy as np
    from npdict import NumpyNDArrayWrappedDict, SparseArrayWrappedDict
    import sys

    # Create dimension labels
    dim1 = [f'item_{i}' for i in range(1000)]
    dim2 = [f'category_{i}' for i in range(1000)]

    # Create dense dictionary
    dense_dict = NumpyNDArrayWrappedDict([dim1, dim2])
    
    # Create sparse dictionary
    sparse_dict = SparseArrayWrappedDict([dim1, dim2])

    # Fill with 1% non-zero elements (10,000 elements)
    for i in range(100):
        for j in range(100):
            dense_dict[f'item_{i}', f'category_{j}'] = 1.0
            sparse_dict[f'item_{i}', f'category_{j}'] = 1.0

    # Compare memory usage
    dense_size = sys.getsizeof(dense_dict.to_numpy())
    sparse_size = sys.getsizeof(sparse_dict.to_dok())
    
    print(f"Dense array size: {dense_size / 1024 / 1024:.2f} MB")
    print(f"Sparse array size: {sparse_size / 1024 / 1024:.2f} MB")

The sparse implementation will typically use significantly less memory in this scenario.