## Project: Predicting Bike-Sharing Patterns

### Goal 

Build a neural network from scratch and use it to predict daily bike rental ridership.
- - - -
### Data

Data comes from the UCI Machine Learning Database. 

The dataset can also be found on Udacity's deep-learning-v2-pytorch github:
https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing/Bike-Sharing-Dataset
- - - -
### Files

* *Predicting_bike_sharing_data.ipynb* - jupyter notebook 
* *Predicting_bike_sharing_data.html* -  html version of jupyter notebook
* *my_answers.py* - python file that defines the neural network
* *requirements.txt* - text file of packages and their versions if using the *pip* command
* *environment.yaml* - yaml file for the conda environment if using Anaconda or Miniconda
- - - -
### Software

This project uses Python 3.7.0 and the following python libraries:

* numpy==1.16.3
* matplotlib==3.0.3
* jupyter notebook==5.7.8
* pandas==0.24.2

All libraries and versions can be found in either *requirements.txt* or *environment.yaml*. 

To install the packages:
`pip install -r /path/to/requirements.txt`

or

`conda env create --name deep-learning --file environment.yaml`

Enter your new environment:
* Mac/Linux: >> `source activate deep-learning`
* Windows: >> `activate deep-learning`
- - - -
### Analysis

Analysis for this project is provided in both *Predicting_bike_sharing_data.ipynb* and *Predicting_bike_sharing_data.html*.

To view *Predicting_bike_sharing_data.html*, use the following link:

http://htmlpreview.github.io/?https://github.com/jonessarae/deep_learning_projects/blob/master/bike_sharing/Predicting_bike_sharing_data.html

To run the jupyter notebook:

`git clone https://github.com/jonessarae/deep_learning_projects.git`  
cd into `bike_sharing` directory  
`jupyter notebook`  
In your browser, open *Predicting_bike_sharing_data.ipynb*.
- - - -
### Considerations for future projects

* Use unit tests to ensure code is free from bugs.
* Include a bias term.
* Apply early stopping to prevent overfitting.
* Incorporate hyperparameter optimization methods.
* Utilize gradual decay of learning rate.

- - - -
### Useful articles

Early stopping:  
https://stats.stackexchange.com/questions/231061/how-to-use-early-stopping-properly-for-training-deep-neural-network

Unit tests:  
https://docs.python-guide.org/writing/tests/

Hyperparameter optimization:  
http://cs231n.github.io/neural-networks-3/#baby


