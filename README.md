# D-Lab's Python Machine Learning Fundamentals Workshop

This repository contains the materials for D-Lab’s Python Machine Learning Fundamentals workshop. Prior experience with [Python Fundamentals](https://github.com/dlab-berkeley/Python-Fundamentals) is assumed.

## Workshop Goals

In this workshop, we provide an introduction to machine learning in Python. First, we'll cover some machine learning basics, including its foundational principles, types of machine learning algorithms, how to fit models, and how to evaluate them. Then, we'll explore several machine learning tasks, includes classification, regression, and clustering. We'll demonstrate how to perform these tasks using `scitkit-learn`, the main package used for machine learning in Python. Finally, we'll go through an automatic model selection tool called `TPOT`.

Basic familiarity with Python is assumed. If you are not comfortable with the material in [Python Fundamentals](https://github.com/dlab-berkeley/Python-Fundamentals), we recommend attending that workshop first.

## Installation Instructions

Anaconda is a powerful package management software that allows you to run Python and Jupyter notebooks very easily. Installing Anaconda is the easiest way to make sure you have all the necessary software to run the materials for this workshop. Complete the following steps:

1. [Download and install Anaconda (Python 3.8 distribution)](https://www.anaconda.com/products/individual). Click "Download" and then click 64-bit "Graphical Installer" for your current operating system.

2. Download the [Python-Machine-Learning-Fundamentals workshop materials](https://github.com/dlab-berkeley/Python-Machine-Learning-Fundamentals):

* Click the green "Code" button in the top right of the repository information.
* Click "Download Zip".
* Extract this file to a folder on your computer where you can easily access it (we recommend Desktop).

3. Optional: if you're familiar with `git`, you can instead clone this repository by opening a terminal and entering `git@github.com:dlab-berkeley/Python-Machine-Learning-Fundamentals.git`.

## Run the code

Now that you have all the required software and materials, you need to run the code:

1. Open the Anaconda Navigator application. You should see the green snake logo appear on your screen. Note that this can take a few minutes to load up the first time. 

2. Click the "Launch" button under "Jupyter Notebooks" and navigate through your file system to the `Python-Machine-Learning-Fundamentals` folder you downloaded above.

3. Click `1_classification.ipynb` to begin.

4. Press Shift + Enter (or Ctrl + Enter) to run a cell.

5. By default, the necessary packages for this workshop should already be installed. You can install them within the Jupyter notebook by running the following line in its own cell:

> ```!pip install seaborn pandas matplotlib numpy jupyter```

Note that all of the above steps can be run from the terminal, if you're familiar with how to interact with Anaconda in that fashion. However, using Anaconda Navigator is the easiest way to get started if this is your first time working with Anaconda.

## Is Python not Working on Your Computer?
If you have a Berkeley CalNet ID, you can run these lessons on UC Berkeley's DataHub by clicking [![Datauhb](https://img.shields.io/badge/launch-datahub-blue)](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fdlab-berkeley%2FPython-Machine-Learning-Fundamentals&urlpath=tree%2FPython-Machine-Learning-Fundamentals%2F&branch=main). By using this link, you can save your work and come back to it at any time. When you want to return to your saved work, just go straight to DataHub [https://datahub.berkeley.edu](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fdlab-berkeley%2FPython-Machine-Learning-Fundamentals&urlpath=tree%2FPython-Machine-Learning-Fundamentals%2F&branch=main), sign in, and you click on the `Python-Machine-Learning-Fundamentals` folder.


If you don't have a Berkeley CalNet ID, you can still run these lessons in the cloud, by clicking this button:
[![Binder](http://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/dlab-berkeley/Python-Machine-Learning-Fundamentals/main?urlpath=tree%2FPython-Machine-Learning-Fundamentals%2F)
By using this button, you cannot save your work unfortunately.

# Additional Resources

Check out the following resources to learn more about machine learning:

* [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html).
* [Stanford's CS229 course materials](https://cs229.stanford.edu/syllabus.html).
* [IBM's free course of machine learning in Python](https://www.edx.org/course/machine-learning-with-python-a-practical-introduct).
* The [Elements of AI course](https://course.elementsofai.com/).

# About the UC Berkeley D-Lab

D-Lab works with Berkeley faculty, research staff, and students to advance data-intensive social science and humanities research. Our goal at D-Lab is to provide practical training, staff support, resources, and space to enable you to use R for your own research applications. Our services cater to all skill levels and no programming, statistical, or computer science backgrounds are necessary. We offer these services in the form of workshops, one-to-one consulting, and working groups that cover a variety of research topics, digital tools, and programming languages.  

Visit the [D-Lab homepage](https://dlab.berkeley.edu/) to learn more about us. You can view our [calendar](https://dlab.berkeley.edu/events/calendar) for upcoming events, learn about how to utilize our [consulting](https://dlab.berkeley.edu/consulting) and [data](https://dlab.berkeley.edu/data) services, and check out upcoming [workshops](https://dlab.berkeley.edu/events/workshops).

# Other D-Lab Python Workshops

Here are other Python workshops offered by the D-Lab:

## Basic competency

* [Python Fundamentals](https://github.com/dlab-berkeley/python-fundamentals)
* [Introduction to Pandas](https://github.com/dlab-berkeley/introduction-to-pandas)
* [Geospatial Fundamentals in Python](https://github.com/dlab-berkeley/Geospatial-Fundamentals-in-Python)
* [Python Visualization](https://github.com/dlab-berkeley/Python-Data-Visualization)

## Intermediate/advanced copmetency

* [Computational Text Analysis in Python](https://github.com/dlab-berkeley/computational-text-analysis-spring-2019)
* [Introduction to Machine Learning in Python](https://github.com/dlab-berkeley/python-machine-learning)
* [Introduction to Artificial Neural Networks in Python](https://github.com/dlab-berkeley/ANN-Fundamentals)
* [Fairness and Bias in Machine Learning](https://github.com/dlab-berkeley/fairML)

# Contributors
* Samy Abdel-Ghaffar
* Sean Perez
* Christopher Hench
