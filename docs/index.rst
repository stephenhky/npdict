Welcome to npdict's documentation!
==================================

.. image:: https://img.shields.io/github/release/stephenhky/npdict.svg?maxAge=3600
   :target: https://github.com/stephenhky/npdict/releases
   
.. image:: https://img.shields.io/pypi/v/npdict.svg?maxAge=3600
   :target: https://pypi.org/project/npdict/
   
.. image:: https://img.shields.io/pypi/dm/npdict.svg?maxAge=2592000&label=installs&color=%2327B1FF
   :target: https://pypi.org/project/npdict/

.. image:: https://readthedocs.org/projects/npdict/badge/?version=latest
   :target: https://npdict.readthedocs.io/en/latest/?badge=latest

``npdict`` is a Python package for dictionary wrappers for NumPy arrays. It aims at facilitating holding numerical
values in a Python dictionary, while retaining the ultra-high performance supported by NumPy.

It supports an object which is a Python dictionary on the surface, but NumPy behind the
back, facilitating fast assignment and retrieval of values and fast computation of NumPy arrays.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   quickstart
   sparse_arrays
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`