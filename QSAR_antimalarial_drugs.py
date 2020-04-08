#!/usr/bin/env python
# coding: utf-8

# # QSAR for anti-malaria drugs
# 
# Dataset was obtained from Chembl database. The project will search for the best QSAR models as a relationship between the input features and the output class (anti-malaria or not). The model input features are obtained as moving averages (MAs) of the original Chembl drug features in specific experimental conditions. The current dataset has already the final MA descriptors. The dataset was cleaned for duplicate rows and the rows where shuffled.
# 
# Input specific libraries for the calculations:

#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')

# remove warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline # Más automático
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight, shuffle

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
#from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif


# Define few global variables (name of class column, the seed for random generator):
outVar = 'Class'
seed = 42 # To ensure that it is always the same split
np.random.seed(seed)

# Read dataset from datasets folder as CSV:
sFile = './datasets/ds.'+str(outVar)+'.csv'
print('\n-> Read dataset', sFile)
df = pd.read_csv(sFile) # To read as DataFrame (data and header)
print('* Columns:',list(df.columns)) # Print the names of the columns to verify
print('* Dimension:', df.shape)