# Feature Engineering Helper for Categorical data 
## Introduction

Feature engineering is the most important step in a general machine learning model building procedure. In machine learning, features refer to the inputs to machine learning models or numerical representations of raw data. There two main types of features in tabular data: numerical feature and categorical feature. Feature engineering is a process of extracting features from raw data and transforming them into suitable formats for the machine learning models. For numerical features, the most frequently used feature engineering methods are log transformation, standardization, and normalization. For categorical features, things are more complicated with not only more options, but when and how to use those methods. Commonly used feature engineering methods for categorical data are ordinal encoding, one-hot encoding, frequency encoding, k-fold target encoding, etc. 
The software is for data science enthusiasts with passion to build their own machine learning models. Feature engineering is the first step required in building a machine learning model. In this step, the software can be used to perform two kinds of tasks for dataset with categorical data: feature encoding and expansion. In the model building process, the software provides a simple way to train multiple models in a pipeline and then display the result of selected models.

## Dependencies
IDE: PyCharm

Platform: MacOS 13.0

Language: Python 3.9

Libraries Used: NumPy, Pandas, Matplotlib, Sklearn, kmodes, plotly

## Software Architecture
There are 5 main modules in the software:
* Encoder
  * Frequency Encoder
  * Target Encoder

* Expander
  * K-prototypes Feature Expander
  
* Visualizer
  * Visualizer for Silhouette diagram
  * Visualizer for model evaluation results

* Trainer
  * Model trainer

* Reporter
  * Model evaluation results reporter
  
## Demo
There is a demo for this package. Most usage of the package is shown in the demo and the doc strings.
