:orphan:

###############
Getting started
###############

Installation
============

Qiskit Algorithms depends on the main Qiskit package which has its own
`Qiskit installation instructions <https://docs.quantum.ibm.com/start/install>`__ detailing the
installation options for Qiskit and its supported environments/platforms. You should refer to
that first, before focusing on the additional installation instructions
specific to Qiskit Algorithms.

Qiskit Algorithms has some functions that have been made optional where the dependent code and/or
support program(s) are not (or cannot be) installed by default.
See :ref:`optional_installs` for more information.

.. tab-set::

    .. tab-item:: Start locally

        The simplest way to get started is to use the pip package manager:

        .. code:: sh

            pip install qiskit-algorithms

    .. tab-item:: Install from source

       Installing Qiskit Algorithms from source allows you to access the most recently
       updated version under development, instead of using the version in the Python Package
       Index (PyPI) repository.

       Since Qiskit Algorithms depends on Qiskit, and its latest changes may require new or changed
       features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions
       `here <https://docs.quantum.ibm.com/start/install-qiskit-source>`__

       .. raw:: html

          <h2>Installing Qiskit Algorithms from Source</h2>

       Using the same development environment that you installed Qiskit in you are ready to install
       Qiskit Algorithms.

       1. Clone the Qiskit Algorithms repository.

          .. code:: sh

             git clone https://github.com/qiskit-community/qiskit-algorithms.git

       2. Cloning the repository creates a local folder called ``qiskit-algorithms``.

          .. code:: sh

             cd qiskit-algorithms

       3. If you want to run tests or linting checks, install the developer requirements.

          .. code:: sh

             pip install -r requirements-dev.txt

       4. Install ``qiskit-algorithms``.

          .. code:: sh

             pip install .

       If you want to install it in editable mode, meaning that code changes to the
       project don't require a reinstall to be applied, you can do this with:

       .. code:: sh

          pip install -e .


.. _optional_installs:

Optional installs
=================

Some optimization algorithms require specific libraries to be run:

* **Scikit-quant**, may be installed using the command `pip install scikit-quant`.

* **SnobFit**, may be installed using the command `pip install SQSnobFit`.

* **NLOpt**, may be installed using the command `pip install nlopt`.


Ready to get going?...
======================

.. qiskit-call-to-action-grid::

    .. qiskit-call-to-action-item::
       :description: Find out about Qiskit Algorithms.
       :header: Dive into the tutorials
       :button_link:  ./tutorials/index.html
       :button_text: Qiskit Algorithms tutorials

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
