Quickstart
==========

This guide will help you get started with ``npdict`` quickly.

Instantiation
------------

Suppose you are doing a similarity dictionary between two sets of words.
And each of these sets have words:

.. code-block:: python

    document1 = ['president', 'computer', 'tree']
    document2 = ['chairman', 'abacus', 'trees']

And you can build a dictionary like this:

.. code-block:: python

    import numpy as np
    from npdict import NumpyNDArrayWrappedDict

    similarity_dict = NumpyNDArrayWrappedDict([document1, document2])

An ``npdict.NumpyNDArrayWrappedDict`` instance is instantiated. It is a Python dict:

.. code-block:: python

    isinstance(similarity_dict, dict)  # which gives `True`

It has a matrix inside with default value 0.0 (and the initial default value can
be changed to other values when the instance is instantiated.)

.. code-block:: python

    similarity_dict.to_numpy()

giving:

.. code-block:: python

    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

Value Assignments
----------------

Now you can assign values just like what you do to a Python dictionary:

.. code-block:: python

    similarity_dict['president', 'chairman'] = 0.9
    similarity_dict['computer', 'abacus'] = 0.7
    similarity_dict['tree', 'trees'] = 0.95

And it has changed the inside numpy array to be:

.. code-block:: python

    array([[0.9  , 0.  , 0.  ],
           [0.  , 0.7 , 0.  ],
           [0.  , 0.  , 0.95]])

Generation of New Object from the Old One
----------------------------------------

If you want to create another dict using the same words, but 
a manipulation of the original value, 25 percent discount
of the original one for example, you can do something like this:

.. code-block:: python

    new_similarity_dict = similarity_dict.generate_dict(similarity_dict.to_numpy()*0.75)

And you got a new dictionary with numpy array to be:

.. code-block:: python

    new_similarity_dict.to_numpy()

giving:

.. code-block:: python

    array([[0.675 , 0.    , 0.    ],
           [0.    , 0.525 , 0.    ],
           [0.    , 0.    , 0.7125]])

This is a simple operation. But the design of this wrapped Python
dictionary is that you can perform any fast or optimized operation
on your numpy array (using numba or Cython, for examples),
while retaining the keywords as your dictionary.

Retrieval of Values
------------------

At the same time, you can set new values just like above, or retrieve
values as if it is a Python dictionary:

.. code-block:: python

    similarity_dict['president', 'chairman']  # returns 0.9

Conversion to a Python Dictionary
--------------------------------

You can also convert this to an ordinary Python dictionary:

.. code-block:: python

    raw_similarity_dict = similarity_dict.to_dict()

Instantiation from a Python Dictionary
-------------------------------------

And you can convert a Python dictionary of this type back to 
``npdict.NumpyNDArrayWrappedDict`` by (recommended):

.. code-block:: python

    new_similarity_dict_2 = NumpyNDArrayWrappedDict.from_dict_given_keywords([document1, document2], raw_similarity_dict)

Or you can even do this (not recommended):

.. code-block:: python

    new_similarity_dict_3 = NumpyNDArrayWrappedDict.from_dict(raw_similarity_dict)

It is not recommended because the order of the keys are not retained in this way.
Use it with caution.