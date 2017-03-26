#**Traffic Sign Recognition Program** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*This is a repository to accomplish the second project of a first term of 
[Udacity - Self-Driving Car Nanodegree](https://www.udacity.com/drive) that started in Feb. 2017.*

## Overview
---

## Prerequisite
This project just uses Python and it's libraries, so you can use any OS to run this project. However, a lot of online advices are based on Linux generally. The choice is yours, but I'm using Ubuntu 16.04. So, if you have any trouble with this project under Ubuntu 16.04, contact me: <hanlsin@gmail.com>

* Python 3
You can download at <https://www.python.org>
I recommend you to use Python 3 for example files in 'example' directory.

### The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Directories and Files
This project contains directories and files as below.

    -- examples[]
    |- src[]
    |- README.md
    |- traffic_sign_recognition_classifier.ipynb
    -- writeup.md

* examples
This is a directory that contains a test dataset.

* traffic_sign_recognition_classifier.ipynb
This file is a Jupyter Notebook. The content is process of a traffic sign recognittion using the LeNet architecture.

* README.md
You are reading this file.

## Test
In this project, there is three test files.

* Python test file
You can execute these test files:
	  $ python test_raw_images.py
	  $ python test_raw_movies.py
* Jupyter Notebook
If you want to Jupiter Notebook, 'find_lane_lines.ipynb', you have to start Jupyter server.
	  $ jupyter notebook
When you execute the command above, your browser is showed up automatically. If not, you can connect manually through this address
	  http://localhost:8888
If you have a problem asking a password, you have to stop the Jupyter server, and follow this <a href="http://jupyter-notebook.readthedocs.io/en/latest/public_server.html">link</a>.
When you connect the Jupyter server, you can find your notebook file, 'find_lane_lines.ipynb', and now you can test.

