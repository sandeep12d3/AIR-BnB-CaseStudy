#!/usr/bin/env python
# coding: utf-8

# ### Lead Score Case Study

# In[1]:


#Assemble libraries

import warnings
warnings.filterwarnings('ignore')
#Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ###### Reading and cleaning data

# In[31]:


# Read and Browse the dataset
lead_data = pd.read_csv(r'C:\Users\ABHIPSA\Desktop\Upgrad\Leads.csv')
lead_data.head()


# In[32]:


# Info of dataset
lead_data.info()


# In[33]:


# describe the dataset
lead_data.shape


# In[34]:


# browsing the dataset
lead_data.head()


# In[35]:


# modifying values to lower case'
lead_data = lead_data.applymap(lambda x:x.lower() if type(x) == str else x)


# In[36]:


# Browsing the dataset now
lead_data.head()


# In[37]:


# select is the default argument on selecting nothing in lead profile so converting all selects to nan
lead_data = lead_data.replace('select',np.nan)


# In[38]:


# Browsing the dataset now
lead_data.head()


# In[39]:


# Checking for unique values in columns
lead_data.nunique()


# In[40]:


# Checking the columns with just 1 unique value
lead_data[['Magazine','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque']]


# In[41]:


# Removing columns with just 1 unique value as these will not add any value to our analysis

lead_data1=lead_data.drop(['Magazine','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'], 1)


# In[42]:


# Browsing the new dataset
lead_data1.head()


# In[44]:


lead_data1.nunique()


# In[45]:


# Check for the percentage of missing values

round(100*(lead_data1.isnull().sum()/len(lead_data1.index)), 2)


# In[58]:


# Removing all columns that have 35% or more null values
lead_data2=lead_data1.drop(['How did you hear about X Education','City','Tags','Lead Quality','Lead Profile','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score'], 1)
lead_data2.head()


# In[59]:


# Check for the percentage of missing values

round(100*(lead_data2.isnull().sum()/len(lead_data2.index)), 2)


# ###### There are columns of null values like country which have a good percentage of null values,however dropping them will
# ###### lead to loss of valuable data as these columns are required. We can replace the nan values with 'not available', this
# ###### will not be of any use, however we can think of dropping them later

# In[60]:


lead_data2['Specialization'] = lead_data2['Specialization'].fillna('not available') 
lead_data2['What matters most to you in choosing a course'] = lead_data2['What matters most to you in choosing a course'].fillna('not available')
lead_data2['Country'] = lead_data2['Country'].fillna('not available')
lead_data2['What is your current occupation'] = lead_data2['What is your current occupation'].fillna('not available')
lead_data2.info()


# In[61]:


# Check for the percentage of missing values

round(100*(lead_data2.isnull().sum()/len(lead_data2.index)), 2)


# In[62]:


# checking for frequency of values in country
lead_data2["Country"].value_counts()


# In[63]:


# changing the categories of country into 3 categories for easier analysis
def country(x):
    group = ""
    if x == "india":
        group = "india"
    elif x == "not available":
        group = "not available"
    else:
        group = "outside india"
    return group

lead_data2['Country'] = lead_data2.apply(lambda x:country(x['Country']), axis = 1)
lead_data2['Country'].value_counts()


# In[64]:


# Check for the percentage of missing values

round(100*(lead_data2.isnull().sum()/len(lead_data2.index)), 2)


# In[65]:


# Checking the percent of lose if the null values are removed
round(100*(sum(lead_data2.isnull().sum(axis=1) > 1)/lead_data2.shape[0]),2)


# In[66]:


lead_data3 = lead_data2[lead_data2.isnull().sum(axis=1) <1]


# In[67]:


# Code for checking number of rows left in percent
round(100*(lead_data3.shape[0])/(lead_data.shape[0]),2)


# In[68]:


# Check for the percentage of missing values

round(100*(lead_data3.isnull().sum()/len(lead_data3.index)), 2)


# In[69]:


# Removing Id values since they are unique for everyone
lead_data_final = lead_data3.drop('Prospect ID',1)
lead_data_final.shape


# ###### EDA

# ###### Univariate Analysis
# ###### Categorical Vehicles

# In[70]:


lead_data_final.info()


# In[75]:


plt.figure(figsize = (25,40))

plt.subplot(8,3,1)
sns.countplot(lead_data_final['Lead Origin'])
plt.title('Lead Origin')

plt.subplot(8,3,2)
sns.countplot(lead_data_final['Do Not Email'])
plt.title('Do Not Email')

plt.subplot(8,3,3)
sns.countplot(lead_data_final['Do Not Call'])
plt.title('Do Not Call')

plt.subplot(8,3,4)
sns.countplot(lead_data_final['Country'])
plt.title('Country')

plt.subplot(8,3,5)
sns.countplot(lead_data_final['Search'])
plt.title('Search')

plt.subplot(8,3,6)
sns.countplot(lead_data_final['Newspaper Article'])
plt.title('Newspaper Article')

plt.subplot(8,3,7)
sns.countplot(lead_data_final['X Education Forums'])
plt.title('X Education Forums')

plt.subplot(8,3,8)
sns.countplot(lead_data_final['Newspaper'])
plt.title('Newspaper')

plt.subplot(8,3,9)
sns.countplot(lead_data_final['Digital Advertisement'])
plt.title('Digital Advertisement')

plt.subplot(8,3,10)
sns.countplot(lead_data_final['Through Recommendations'])
plt.title('Through Recommendations')

plt.subplot(8,3,11)
sns.countplot(lead_data_final['A free copy of Mastering The Interview'])
plt.title('A free copy of Mastering The Interview')

plt.subplot(8,3,12)
sns.countplot(lead_data_final['Last Notable Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')


plt.show()


# In[76]:


sns.countplot(lead_data_final['Lead Source']).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[77]:


plt.figure(figsize = (25,35))
plt.subplot(2,2,1)
sns.countplot(lead_data_final['Specialization']).tick_params(axis='x', rotation = 90)
plt.title('Specialization')
plt.subplot(2,2,2)
sns.countplot(lead_data_final['What is your current occupation']).tick_params(axis='x', rotation = 90)
plt.title('Current Occupation')
plt.subplot(2,2,3)
sns.countplot(lead_data_final['What matters most to you in choosing a course']).tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')
plt.subplot(2,2,4)
sns.countplot(lead_data_final['Last Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')
plt.show()


# In[78]:


sns.countplot(lead_data['Converted'])
plt.title('Converted("Y variable")')
plt.show()


# ###### Numerical Variables
# 

# In[79]:


lead_data_final.info()


# In[80]:


plt.figure(figsize = (15,15))
plt.subplot(321)
plt.hist(lead_data_final['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(322)
plt.hist(lead_data_final['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(323)
plt.hist(lead_data_final['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


# In[82]:


# Showing all categorical variables to converted

plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(1,2,2)
sns.countplot(x='Lead Source', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[84]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Do Not Email', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 45)
plt.title('Do Not Email')

plt.subplot(1,2,2)
sns.countplot(x='Do Not Call', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 45)
plt.title('Do Not Call')
plt.show()


# In[87]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Last Activity', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')

plt.subplot(1,2,2)
sns.countplot(x='Country', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Country')
plt.show()


# In[88]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()


# In[89]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='What matters most to you in choosing a course', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')

plt.subplot(1,2,2)
sns.countplot(x='Search', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Search')
plt.show()


# In[90]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper Article', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Newspaper Article')

plt.subplot(1,2,2)
sns.countplot(x='X Education Forums', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('X Education Forums')
plt.show()


# In[91]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Newspaper')

plt.subplot(1,2,2)
sns.countplot(x='Digital Advertisement', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Digital Advertisement')
plt.show()


# In[92]:


plt.figure(figsize = (15,10))

plt.subplot(1,2,1)
sns.countplot(x='Through Recommendations', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Through Recommendations')

plt.subplot(1,2,2)
sns.countplot(x='A free copy of Mastering The Interview', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('A free copy of Mastering The Interview')
plt.show()


# In[93]:


sns.countplot(x='Last Notable Activity', hue='Converted', data= lead_data_final).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')
plt.show()


# In[94]:


# Check the correlation among variables
plt.figure(figsize=(20,15))
sns.heatmap(lead_data_final.corr())
plt.show()


# ###### From the EDA we can discern that there are many variables with limited data so analysis on this will be defferred

# In[95]:


numeric = lead_data_final[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# ##### The outliers are not much and well contained so moving on to analysis

# ### Dummy Variables

# In[96]:


lead_data_final.info()


# In[156]:


lead_data_final.loc[:, lead_data_final.dtypes == 'object'].columns


# In[194]:


# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(lead_data_final[['Lead Origin','Specialization' ,'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)
# Add the results 
lead_data_final_dummy = pd.concat([lead_data_final, dummy], axis=1)
lead_data_final_dummy


# In[195]:


lead_data_final_dummy = lead_data_final_dummy.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call','Last Activity', 'Country', 'Specialization', 'What is your current occupation','What matters most to you in choosing a course', 'Search','Newspaper Article', 'X Education Forums', 'Newspaper','Digital Advertisement', 'Through Recommendations','A free copy of Mastering The Interview', 'Last Notable Activity','Last Notable Activity_view in browser link clicked'], 1)
lead_data_final_dummy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### TEST-TRAIN SPLIT

# In[196]:


# Assemble sklearn
from sklearn.model_selection import train_test_split


# In[197]:


X = lead_data_final_dummy.drop(['Converted'], 1)
X.head()


# In[198]:


# Putting the target variable in y
y = lead_data_final['Converted']
y.head()


# In[199]:


# Split the dataset into 70% and 30% for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)


# In[200]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# In[ ]:





# In[ ]:





# In[202]:


# Import MinMax scaler
from sklearn.preprocessing import MinMaxScaler
# Scale the three numeric features
scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# In[ ]:





# In[203]:


# To check the correlation among varibles
plt.figure(figsize=(20,30))
sns.heatmap(X_train.corr())
plt.show()


# ### Model Building

# In[204]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[205]:


rfe.support_


# In[188]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[206]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[207]:


X_train.columns[~rfe.support_]


# In[208]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[211]:


# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[212]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ###### p-value of Last Notable Activity_had a phone conversation is high so dropping the column

# In[213]:


col = col.drop('Last Notable Activity_had a phone conversation',1)


# In[214]:


#BUILDING MODEL #2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[215]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ###### p-value of What is your current occupation_housewife is high so dropping the column vifs being stable

# In[216]:


col = col.drop('What is your current occupation_housewife',1)


# In[218]:


#BUILDING MODEL #3
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[219]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ###### So the Values all seem to be in order so now, Moving on to derive the Probabilities, Lead Score, Predictions on Train

# In[220]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[221]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[222]:


# Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[223]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[224]:


# Importing metrics from sklearn for evaluation
from sklearn import metrics


# In[225]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[ ]:


# Predicted     not_churn    churn
# Actual
# not_churn        3438       457
# churn             737      1719


# In[226]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# ###### 81% accuracy good for our model

# In[227]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[228]:


# Calculating the sensitivity
TP/(TP+FN)


# In[229]:


# Calculating the specificity
TN/(TN+FP)


# ###### Taking the current cutoff at 0.5 we have about 70% sensitivity,88% specificity and 81% accuracy

# ### Optimize cutoff ROC curve

# In[230]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[231]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[232]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# ###### the area under the ROC is 0.88 which is good for our model

# In[233]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[234]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[235]:


# Plotting the 3 points
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# ###### from the graph it is visible that the optimal cutoff is at 0.38

# In[242]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.38 else 0)
y_train_pred_final.head()


# In[243]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[244]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[245]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[246]:


# Calculating the sensitivity
TP/(TP+FN)


# In[247]:


# Calculating the specificity
TN/(TN+FP)


# ###### with the cut off at 0.38 we have accuracy and specifity at 80% while sensitivity is 78%

# ### Prediction on Test set

# In[248]:


# Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[249]:


# Select the columns in X_train for X_test as well
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm


# In[250]:


#Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[256]:


# Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.38 else 0)
y_pred_final


# In[255]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[257]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[258]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[259]:


# Calculating the sensitivity
TP/(TP+FN)


# In[ ]:





# In[261]:


####### With the current cut off as 0.35 we have accuracy is around 46%, sensitivity is 99% and that is a good tradeoff


# ### Precision Recall

# In[262]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[263]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[264]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ###### With the current cut off as 0.38 we have Precision around 78% and Recall around 70%

# ### Precision and recall tradeoff

# In[265]:


from sklearn.metrics import precision_recall_curve


# In[266]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[267]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[268]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[269]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_train_pred_final.head()


# In[270]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[271]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[272]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[273]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[274]:


#Recall = TP / TP + FN
TP / (TP + FN)


# ###### With the current cut off as 0.41 we have Precision around 75% and Recall around 76%

# ### Prediction on Test set

# In[275]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[276]:


# Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_pred_final


# In[277]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[278]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[279]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[ ]:





# In[281]:


#Recall = TP / TP + FN
TP / (TP + FN)


# ###### With the current cut off as 0.41 we have Precision around 47% and Recall around 75%

# ####### Conclusion
# It was found that the variables that mattered the most in the potential buyers are (In descending order) :
# 
# The total time spend on the Website.
# Total number of visits.
# When the lead source was:
# a. Google
# b. Direct traffic
# c. Organic search
# d. Welingak website
# 4. When the last activity was:
# a. SMS
# b. Olark chat conversation
# 5. When the lead origin is Lead add format. 6. When their current occupation is as a working professional.
# Keeping these in mind the X Education can flourish as they have a very high chance to get almost all the potential buyers to change their mind and buy their courses.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




