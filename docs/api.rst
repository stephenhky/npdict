API Reference
============

This page provides detailed API documentation for all public classes and functions in the ``npdict`` package.

NumpyNDArrayWrappedDict
----------------------

.. autoclass:: npdict.wrap.NumpyNDArrayWrappedDict
   :members:
   :special-members: __init__, __getitem__, __setitem__, __iter__, __repr__, __str__, __len__
   :undoc-members:
   :show-inheritance:

SparseArrayWrappedDict
---------------------

.. autoclass:: npdict.sparse.SparseArrayWrappedDict
   :members:
   :special-members: __init__, __getitem__, __setitem__, __repr__
   :undoc-members:
   :show-inheritance:

Exceptions
---------

.. autoclass:: npdict.utils.DuplicatedKeyError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: npdict.utils.WrongArrayDimensionException
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: npdict.utils.WrongArrayShapeException
   :members:
   :undoc-members:
   :show-inheritance: