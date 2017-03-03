# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:45:37 2017

@author: siowmeng
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

loans = pd.read_excel('LCloanbook.xls', sheetname = 'Data')

x = loans.iloc[ : , 1:].as_matrix()
y = loans.iloc[ :, 0].as_matrix()

# Standardise before using Logit with Regularization
x = StandardScaler().fit_transform(x)

# Logit with 10-fold cross-validation
LRCV_l1 = LogisticRegressionCV(Cs = [0.002], 
                               cv = KFold(10, shuffle = True, random_state = 99), 
                                         penalty='l1', solver = 'liblinear')
LRCV_l1.fit(x, y)

print("Number of Attributes after 10-fold cross-validation: ", sum(LRCV_l1.coef_[0] != 0))
print("Average accuracy over 10-fold cross-validation: ", np.mean(LRCV_l1.scores_[1]))
print("In-Sample Accuracy=", LRCV_l1.score(x, y))

# Logit using all data without cross-validation
for c in np.arange(0.0015, 0.0025, 0.0001):
    LR_l1 = LogisticRegression(C = c, penalty='l1')
    LR_l1.fit(x, y)
    
    print("C=", c)
    print("Number of Attributes=", sum(LR_l1.coef_[0] != 0))
    print("In-Sample Accuracy=", LR_l1.score(x, y))
    
