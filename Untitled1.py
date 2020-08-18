#!/usr/bin/env python
# coding: utf-8

# <b>TASK 4
# 
# 
# To Explore the Decision Tree Algorithm</b>

# In[6]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# In[40]:


iris = load_iris()


# In[41]:



X = iris.data
y = iris.target


# <b>Lets train the Decision Tree Classifier to our iris dataset<b/>

# In[42]:



from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=0)

clf.fit(X,y)


# <b>Let's Visualize the Decision Tree Classifier</b>

# In[10]:



import matplotlib.pyplot as plt

plt.figure(figsize=(20,30))
tree.plot_tree(clf,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
plt.show()


# In[ ]:




