# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This projects demonstrates how to implement a machine learning model to predict customer churn probability. The project follows clean code best practices and includes modular code that can be run from command line or interactively and follows PEP8 guidelines

data artifacts stored in /data/ folder
image artifacts stored in /images/ folder
log file stored in /logs/ folder

## Files and data description
churn_library.py - file contains functions to implement machine learning model. Functions in this file include:
    - Load data
    - Feature engineering
    - Model training and evaluation

churn_script_logging_and_tests.py - this file contains test scripts and logging of key artifacts and success or failure messages

churn_notebook.ipynb - jupyter notebook with all code models from churn_library.py - used for initial developement

Guide.ipynb - guide to completing the project

README.md - this file


## Running Files
to run the complete project with all testing and logging run the churn_script_logging_and_tests.py file in python session in terminal as: 
    "python churn_script_logging_and_tests.py"
    
to run certain tests: edit the if __name__ == '__main__'
section to remove the modules you do not want to run.

To run the machine learning code without logging or testing, you can run the 'churn_library.py' file in the same way as above.


## Dependencies
scikit-learn==0.22       
shap==0.40.0     
joblib==0.11
pandas==0.23.3
numpy==1.19.5 
matplotlib==2.1.0      
seaborn==0.11.2
pylint==2.7.4
autopep8==1.5.6



