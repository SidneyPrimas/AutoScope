Description: Readme that describes the appropriate setup for a virtualenv with tensorflow (and matplotlib GUI functionality). 

1) Install a virtualenv with --system-site-packages. With this flage, the virtualenv will inherent the site pages from the global site package directory (usually /usr/lib/python2.7/site-packages). --system-site-packages is recommended by TF to ensure less issues and more ease of use. We should be able to implement a more segregated version as well. 

2) Install tensorflow locally. Use: pip install --upgrade tensorflow. This will install tensorflow within the local virtualenv. 

3) Install the necessary libraries with pip install --upgrade X. For example: pip install --upgrade numpy scipy matplotlib pandas
Note: Some of these will already exist in the global site packages. However, we want to use the local site packages and so need to reinstall these locally. 

4) To show matplitlib figures, we need to give the virtualenv to features external to the environment. This is done with the frameworkpython command. Include the below in a framewhorkpython bash command. This provides access to external packages, but prioritizes packages within the environment. 
Link: https://matplotlib.org/faq/osx_framework.html#osxframework-faq
#!/bin/bash

# what real Python executable to use
PYVER=2.7
PATHTOPYTHON=/usr/bin/
PYTHON=${PATHTOPYTHON}python${PYVER}

# find the root of the virtualenv, it should be the parent of the dir this script is in
ENV=`$PYTHON -c "import os; print(os.path.abspath(os.path.join(os.path.dirname(\"$0\"), '..')))"`

# now run Python with the virtualenv set as Python's HOME
export PYTHONHOME=$ENV
exec $PYTHON "$@"


# To optimize Tensorflow for your CPU, need to install a custom TF.
Follow this tutorial: http://mortada.net/tips-for-running-tensorflow-with-gpu-support-on-aws.html