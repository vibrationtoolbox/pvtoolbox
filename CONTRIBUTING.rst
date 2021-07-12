Reporting bugs
--------------
If you find a bug, please open an issue on the `Github issues tracker <https://github.com/vibrationtoolbox/vibration_toolbox/issues>`_.
Please provide some code that reproduces the error and versions of the packages installed.

Contributing code
-----------------
To contribute code we recommend you follow these steps:

1. Fork the repository on github

2. Set up travis-ci for your fork. This is actually pretty quick and easy:

  #. Go to travis-ci.org and Sign in with GitHub.

  #. Account page will show all the repositories attached to GitHub.

  #. Find the right repository and enable Travis CI.

  #. Once this is done, Travis CI will be turned-on in GitHub fork.

  #. Go back to the fork on GitHub, click    

     ``Settings`` -> ``Webhooks`` -> ``updated travis-ci.org link``.

  #. Default or customize the options based on needs and click 

     ``Update webhook``. 

3. Clone the repository to your favorite location on your drive where you want to work on it.

4. Before installing in developer mode, please be sure to ``pip uninstall vibration_toolbox`` if it is already installed.

5. To work in `developer mode <https://packaging.python.org/distributing/#working-in-development-mode>`_, at the top level directory inside the ``vibration toolbox`` type::

    $ pip install -e .

   This will allow you to edit the code while having it pretend to be installed. Keep in mind, if you have actually installed the ``vibration toolbox`` you may have a conflict. You must uninstall it and install your development version with the command above.[*]_

6. Add your code/make your modifications, committing to your branch.

  #. On some applications you need to stage then commit.

  #. After committing, some applications will require you to do a push-pull

    #. The pull makes sure that what's on your computer is up to date with what's in your fork on GitHub.

    #. The push moves your changes to the GitHub repository.

7. If a new function is added
   please provide docstrings following the `Numpy standards for docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
   The docstrings should contain examples to be tested.

   Specifically note:

   1. Parameters should be listed similarly to:

    |    filename : str
    |    copy : bool
    |    dtype : data-type
    |    iterable : iterable object
    |    shape : int or tuple of int
    |    files : list of str
    |    time : array_like

   2. First line should be inline with the ``"""`` and brief enough to fit on one line.

   3. There must be a blank line after the first line.

   This is not exhaustive. It just highlights some consistent errors made.

8. Run the doctests regularly when you make edits.

   To run the doctests, `pytest <https://docs.pytest.org/en/latest/>`_ is needed and can be installed with ``pip install -U pytest``.

   To run the tests from the shell you can `cd` to the project's root directory and type::

     $ pytest

    1. To run the tests from ``pycharm`` you can do: Run -> Edit Configurations -> Add -> python tests -> pytest Then just set the path to the project directory.

    2. To run the tests from ``spyder`` see `spyder-unittest <https://github.com/spyder-ide/spyder-unittest`_.

9. Commit and check `travis-ci <https://travis-ci.org/vibrationtoolbox/vibration_toolbox>`_ tests regularly. Having a great number of changes before a commit can make tracing errors very hard. Make sure you are looking at your branch when assessing whether it's working.

10. `Update from the main repository <https://www.sitepoint.com/quick-tip-sync-your-fork-with-the-original-without-the-cli/>`_ regularly, and certainly before submitting a pull request. This allows you to see the complete results before we look at them.  If it doesn't work, the pull will (should) be denied.

Alternatively, from the command line at the top directory within the repository:

.. code-block:: bash

  git pull origin master

`Reconcile any merge conflicts`_. Basically, you will edit the files that it complains about to choose which edits win and which edits lose.

After this push commits to your repository (most likely using GitHub Desktop) and **then** submit a pull request if it passes the TravisCI tests. 

11. If the tests are passing, make a git pull (in your GitHub app) to assure that your code is up to date with your master branch and that your code has no conflicts with the current base. Doing this regularly ensures that your accumulated edits won't be massively in conflict with the existing code base. After that, push your branch to GitHub and then open a pull request on GitHub.

Instructions bellow are directed to main developers
===================================================

To make distribution and release
--------------------------------

1) Edit the version number in ``vibration_toolbox/__init__.py``
2) Use the Makefile, ``make release``

The ``conf.py`` file for the documentation pulls the version from ``__init__.py``

To make a distribition (for testing or posting to github)
-----------------------------------------------------------

.. code-block:: bash

  >> make wheel

To test before release
----------------------

Check the Travis CI logs. They are more comprehensive.

To test distribution installabilty
-----------------------------------
Note: these are out of date and saved only for historical reasons.

python setup.py register -r pypitest
python setup.py sdist upload -r pypitest

look at https://testpypi.python.org/pypi

Other information sites
------------------------

`twine notes <https://packaging.python.org/distributing/#working-in-development-mode>`_

https://pypi.python.org/pypi/wheel

.. [*] The top level directory contains ``CONTRIBUTING.rst``, ``LICENSE.txt``, ``requirements.txt``, etc.

.. _`Reconcile any merge conflicts`:
https://help.github.com/articles/addressing-merge-conflicts/
