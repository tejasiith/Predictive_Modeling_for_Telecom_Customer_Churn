#!/usr/bin/env python
# coding: utf-8

# # Multivariate Logistic Regression Project

# ## Telecom Churn Case Study
# ### Problem Statement
# #### You have a telecom firm which has collected data of all its customers. The main types of attributes are:
# #### 1. Demographics (age, gender etc.)
# #### 2. Services availed (internet packs purchased, special offers taken, etc.)
# #### 3. Expenses (amount of recharge done per month etc.)
# ####   Based on all this past information, you want to build a model which will predict whether a particular customer will churn or not, i.e. whether they will switch to a different service provider or not.

# ####  So the variable of interest, i.e. the target variable here is ‘Churn’ which will tell us whether or not a particular customer has churned. It is a binary variable: 1 means that the customer has churned and 0 means the customer has not churned.

# ## Step 1:- Importing and Merging the data

# In[445]:


# suppressing warnings
import warnings
warnings.filterwarnings('ignore')


# In[446]:


# importing pandas and numpy
import pandas as pd, numpy as np


# In[447]:


# importing all dataset
churn_data= pd.read_csv(r'C:\Users\DELL\Documents\PROJECTS\Multiple_Logistic_Regression\churn_data.csv')
churn_data.head()


# In[448]:


customer_data= pd.read_csv(r'C:\Users\DELL\Documents\PROJECTS\Multiple_Logistic_Regression\customer_data.csv')
customer_data.head()


# In[449]:


internet_data= pd.read_csv(r'C:\Users\DELL\Documents\PROJECTS\Multiple_Logistic_Regression\internet_data.csv')
internet_data.head()


# ## Combining all data files into one consollidated dataframe

# In[450]:


# Merging on 'customerID'
df_1=pd.merge(churn_data, customer_data, how= 'inner', on='customerID')


# In[451]:


telecom=pd.merge(df_1, internet_data,how= 'inner', on='customerID')


# ## Step 2:- Inspecting the Dataframe

# In[452]:


# Let's see the head of our master dataset
telecom.head()


# In[453]:


# Let's check the dimensions of Dataframe
telecom.shape


# In[454]:


# Let's look at the statistical aspects of Dataframe
telecom.describe()


# In[455]:


# Let's see the type of each column
telecom.info()


# ## Step 3:- Data preparation

# ### Converting some binary variables to (Yes/No) to (0/1)

# In[456]:


# List variables to map
varlist=['PhoneService','PaperlessBilling','Churn','Partner','Dependents']

# Defining the map function 
def binary_map(x):
    return x.map({"Yes":1,"No":0})

# Apply the function to the housing list
telecom[varlist] = telecom[varlist].apply(binary_map)


# In[457]:


telecom.head()


# ### For categorrical variables with multiple levels, create dummy features(one-hot encoded)

# In[458]:


# Creating a dummy variable for some of the categorical variables and dropping the first one

dummy1= pd.get_dummies(telecom[['Contract', 'PaymentMethod','gender','InternetService']],drop_first=True)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,dummy1],axis=1)

telecom.head()


# In[459]:


# Creating dummy variables for the reamaining categorical variables and dropping the parent variable

# Creating dummy variables for the variable 'MultipleLines'
ml=pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')

# Dropping MultipleLines_No phone service column
ml1=ml.drop(['MultipleLines_No phone service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,ml1],axis=1)



# Creating dummy variables for the variable 'OnlineSecurity'
os=pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')

# Dropping OnlineSecurity_No internet service column
os1=os.drop(['OnlineSecurity_No internet service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,os1],axis=1)



# Creating dummy variables for the variable 'OnlineBackup'
ob=pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')

# Dropping OnlineBackup_No internet service column
ob1=ob.drop(['OnlineBackup_No internet service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,ob1],axis=1)



# Creating dummy variables for the variable 'DeviceProtection'
dp=pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')

# Dropping DeviceProtection_No internet service column
dp1=dp.drop(['DeviceProtection_No internet service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,dp1],axis=1)



# Creating dummy variables for the variable 'TechSupport'
ts=pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')

# Dropping TechSupport_No internet service column
ts1=ts.drop(['TechSupport_No internet service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,ts1],axis=1)



# Creating dummy variables for the variable 'StreamingTV'
st=pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')

# Dropping StreamingTV_No internet service column
st1=st.drop(['StreamingTV_No internet service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,st1],axis=1)



# Creating dummy variables for the variable 'StreamingMovies'
sm=pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')

# Dropping MultipleLines_No internet service column
sm1=sm.drop(['StreamingMovies_No internet service'],1)

# Adding the results to the master dataframe
telecom=pd.concat([telecom,sm1],axis=1)


# In[460]:


telecom.head()


# ## Dropping the repeated variables

# In[461]:


# We have created dummies for below variables, so we can drop them
# List of columns to drop
columns_to_drop = ['Contract', 'PaymentMethod', 'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Drop columns if they exist
telecom = telecom.drop(columns_to_drop, axis=1, errors='ignore')

# Convert 'TotalCharges' to float
telecom['TotalCharges'] = pd.to_numeric(telecom['TotalCharges'], errors='coerce')


# In[462]:


telecom.info()


# #### Now you can see that you have all data in numeric

# ### Checking for outliers

# In[463]:


# Checking for outliers in the continuous variable
num_telecom=telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[464]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_telecom.describe(percentiles=[.25,.50,.75,.90,.95,.99])


# #### From destribution shown in above, you can see that there no outliers in your data. The numbers are gradually increasing

# ### Checking for Missing values and Inputing them

# In[465]:


# Adding up the missing values(Column_wise)
telecom.isnull().sum()


# In[466]:


# Checking the percent of missing values
round(100*(telecom.isnull().sum()/len(telecom.index)),2)


# In[467]:


# Removing NaN TotalCharges rows
telecom=telecom[~np.isnan(telecom['TotalCharges'])]


# In[468]:


# Checking percentage of missing values after removing the missing values
round(100*(telecom.isnull().sum()/len(telecom.index)),2)


# ## Step 4:- Test-Train Split

# In[469]:


from sklearn.model_selection import train_test_split


# In[470]:


# Putting features variables to x
x=telecom.drop(['Churn','customerID'],axis=1)
x.head()


# In[471]:


# Putting response variable to y
y=telecom['Churn']
y.head()


# In[472]:


# Splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3)


# ## Step 5:- Feature Scaling

# In[473]:


from sklearn.preprocessing import StandardScaler


# In[603]:


scaler=StandardScaler()

x_train[['tenure']]=scaler.fit_transform(x_train[['tenure']])
x_train.head()


# In[475]:


# Checking the churn rate
churn=(sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn


# #### We have 26.58% churn rate

# ## Step 6:- Looking at correlations

# In[476]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[477]:


# Let's see the correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(telecom.corr(),annot=True)
plt.show()


# In[478]:


#print(telecom.columns)

# Droping highly correlated dummy variables
#x_test=x_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)
#x_train=x_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)


# ### Checking the correlation matrix

# #### After dropping highly correlated variables, now let's check the correlation matrix

# In[479]:


plt.figure(figsize=(20,10))
sns.heatmap(x_train.corr(),annot=True)
plt.show()


# ## Step 7:- Model Building
# ### Let's start by splitting our data into training set and a test set

# In[480]:


# Running our first Training Model
import statsmodels.api as sm


# In[481]:


# Logistic regression model
logml=sm.GLM(y_train,(sm.add_constant(x_train)),family=sm.families.Binomial())
logml.fit().summary()


# ## Step 8:- Feature Selection Using RFE

# In[482]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[483]:


from sklearn.feature_selection import RFE

rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe = rfe.fit(x_train, y_train)


# In[484]:


rfe.support_


# In[485]:


list(zip(x_train.columns, rfe.support_,rfe.ranking_))


# In[486]:


col=x_train.columns[rfe.support_]


# In[487]:


x_train.columns[~rfe.support_]


# #### Accessing the model with Statsmodel

# In[488]:


x_train_sm=sm.add_constant(x_train[col])
logm2=sm.GLM(y_train,x_train_sm, family=sm.families.Binomial())
res=logm2.fit()
res.summary()


# In[489]:


# Getting the predicted values on the train set
y_train_pred=res.predict(x_train_sm)
y_train_pred[:10]


# In[490]:


y_train_pred=y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating a dataframe with a actual churm flag and the predicted probabilities

# In[491]:


y_train_pred_final=pd.DataFrame({'Churn':y_train.values,'Churn_Prob':y_train_pred})
y_train_pred_final['custID']=y_train.index
y_train_pred_final.head()


# #### Creating new column 'Predicted' with 1 if Churn_Prob>0.5 else  0

# In[492]:


y_train_pred_final['predicted']=y_train_pred_final.Churn_Prob.map(lambda x: 1 if x>0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[493]:


from sklearn import metrics


# In[494]:


# Confusion metrics
confusion=metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.predicted)
print(confusion)


# In[495]:


# Predicted     not_churn   churn
# Actual
# not_churn      3236        379
# churn          578         729


# In[496]:


# Let's check the overall accuracy
print(metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted))


# ### Checking VIFs

# In[497]:


# Check for the VIF values of the feature variables
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[498]:


# Create a DataFrame that will contain the names of all the feature variables 
vif= pd.DataFrame()
vif['Features']= x_train[col].columns
vif['VIF']= [variance_inflation_factor(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif['VIF']= round(vif['VIF'],2)
vif= vif.sort_values(by= "VIF", ascending=False)
vif


# ##### There are few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making th model complex. The variable 'PhoneService' has a high VIF. So, let's start by dropping that.

# In[499]:


col = col.drop('MonthlyCharges', 1)
col


# In[500]:


# Let's re-run the model using the slected variables
x_train_sm=sm.add_constant(x_train[col])
logm3=sm.GLM(y_train,x_train_sm, family=sm.families.Binomial())
res=logm3.fit()
res.summary()


# In[501]:


y_train_pred=res.predict(x_train_sm).values.reshape(-1)


# In[502]:


y_train_pred[:10]


# In[503]:


y_train_pred_final['ChurnProb']=y_train_pred


# In[504]:


# Creating new column 'Predicted' with 1 if Churn_Prob>0.5 else  0
y_train_pred_final['predicted']=y_train_pred_final.ChurnProb.map(lambda x: 1 if x>0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[505]:


# Let's check the overall accuracy
print(metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted))


# ### Check for the VIF values

# In[506]:


vif= pd.DataFrame()
vif['Features']= x_train[col].columns
vif['VIF']= [variance_inflation_factor(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif['VIF']= round(vif['VIF'],2)
vif= vif.sort_values(by= "VIF", ascending=False)
vif


# In[507]:


# Let's drop 'TotalCharges' since it has high VIF values
col = col.drop('TotalCharges', 1)
col


# In[508]:


# Let's re-run the model using the slected variables
x_train_sm1=sm.add_constant(x_train[col])
logm3=sm.GLM(y_train,x_train_sm1, family=sm.families.Binomial())
res=logm3.fit()
res.summary()


# In[509]:


y_train_pred=res.predict(x_train_sm1).values.reshape(-1)


# In[510]:


y_train_pred[:10]


# In[511]:


y_train_pred_final['ChurnProb']=y_train_pred


# In[512]:


# Creating new column 'Predicted' with 1 if Churn_Prob>0.5 else  0
y_train_pred_final['predicted']=y_train_pred_final.ChurnProb.map(lambda x: 1 if x>0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[513]:


# Let's check the overall accuracy
print(metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted))


# In[514]:


vif= pd.DataFrame()
vif['Features']= x_train[col].columns
vif['VIF']= [variance_inflation_factor(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif['VIF']= round(vif['VIF'],2)
vif= vif.sort_values(by= "VIF", ascending=False)
vif


# ##### All variables have a good value of VIF. So we need not any more variables and we can proceed with making predictions using this model only.

# In[515]:


# Let's take a look at Confusion metrics again
confusion1=metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.predicted)
print(confusion1)


# In[516]:


# Predicted     not_churn   churn
# Actual
# not_churn      3274        358
# churn          595         695


# In[517]:


# Let's check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted)


# ### Metrics beyond simply accuracy

# In[518]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[519]:


# Let's see the sensitivity of our logistic regression model
TP/float(TP+FN)


# In[520]:


# Let's see the specificity of our logistic regression model
TN/float(TN+FP)


# In[521]:


# Calculate false pasitive rate-predicting churn when customer does not have churn
FP/float(FP+TN)


# In[522]:


# Positive predictuve value
TP/float(TP+FP)


# In[523]:


# Negative predictuve value
TN/float(TN+FN)


# ### Step 9:- Plotting the ROC Curve
# 
# #### An ROC Curve demonstrates a several thins:
# ####       1. It shows the tradeoff between sensitivity and specificity(any increase in sensitivity wii be accompanied by a decrease in specificity).
# ####       2. The closer the curve follows left-hand border and then the top border of ROC space, the more accurate the test.
# ####       3. The closer the curve comes to the 45 degree diagonal of the ROC space, the less accurate the test.

# In[524]:


# Defining the function to plot the ROC curve
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

# Calling the function
draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[525]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[526]:


draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# ### Finding Optimal Cutoff Point
# #### Optimal Cutoff probability is that where we get balanced sensitivity and specificity

# In[527]:


# Let's create a columns with different probability cutoffs
numbers=[float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]=y_train_pred_final.Churn_Prob.map(lambda x:1 if x>i else 0)
y_train_pred_final.head()


# In[528]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[529]:


# Let's plot Accuracy, Sensitivity and specificity for various probabilities
cutoff_df.plot.line(x='prob',y=['accuracy', 'sensi','speci'])
plt.show()


# #### From the curve above,0.3 is the optimum point to take it as a cutoff probability

# In[530]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[531]:


# Let's check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.final_predicted)


# ### Precision and Recall

# In[590]:


# Looking at confusion matrix again
confusion= metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.final_predicted)
confusion


# ### Precision

# In[533]:


#TP/TP+FP
confusion[1,1]/(confusion[1,1]+confusion[0,1])


# ### Recall

# In[534]:


#TP/TP+FN
confusion[1,1]/(confusion[1,1]+confusion[1,0])


# ##### Using sklearn utilities for the same

# In[535]:


from sklearn.metrics import precision_score, recall_score


# In[536]:


get_ipython().run_line_magic('pinfo', 'precision_score')


# In[537]:


precision_score(y_train_pred_final.Churn,y_train_pred_final.final_predicted)


# In[538]:


recall_score(y_train_pred_final.Churn,y_train_pred_final.final_predicted)


# #### Precision and Recall tradeoff

# In[539]:


from sklearn.metrics import precision_recall_curve


# In[540]:


y_train_pred_final.Churn,y_train_pred_final.final_predicted


# In[541]:


p,r,thresholds=precision_recall_curve(y_train_pred_final.Churn,y_train_pred_final.final_predicted)


# In[542]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Step 11:- Making Predictions on the test set

# In[606]:


x_test[['tenure']] = scaler.transform(x_test[['tenure']])


# In[631]:


x_test=x_test[col]
x_test.head()


# In[632]:


x_test_sm=sm.add_constant(x_test)


# Making predictions on test set

# In[633]:


y_test_pred=res.predict(x_test_sm)


# In[634]:


y_test_pred[:10]


# In[636]:


# Converting y_pred to a dataframe which is an array
y_pred= pd.DataFrame(y_test_pred)


# In[637]:


# Let's see the head
y_pred.head()


# In[613]:


# Coverting y_test to dataframe
y_test_df= pd.DataFrame(y_test)


# In[614]:


# Putting CustID to index
y_test_df['CustID']=y_test_df.index


# In[615]:


# Removing index for both DataFrame to append them side by side
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[638]:


# Appending y_test_df and y_pred_1
y_pred_final=pd.concat([y_test_df, y_pred_1],axis=1)


# In[639]:


y_pred_final.head()


# In[643]:


# Renaming the column
y_pred_final= y_pred_final.rename(columns={ 0 :'Churn_Prob'})


# In[647]:


# Rearranging the columns
#y_pred_final= y_pred_final.reindex(['CustID','Churn','Churn_Prob'],axis=1)


# In[654]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[655]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x : 1 if x>0.3 else 0)


# In[656]:


y_pred_final.head()


# In[662]:


# Let's check the overall accuracy
metrics.accuracy_score(y_pred_final.Churn,y_pred_final.final_predicted)


# In[664]:


# Looking at confusion matrix again
confusion2= metrics.confusion_matrix(y_pred_final.Churn,y_pred_final.final_predicted)
confusion2


# In[665]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[666]:


# Let's see the sensitivity of our logistic regression model
TP/float(TP+FN)


# In[667]:


# Let's see the specificity of our logistic regression model
TN/float(TN+FP)


# In[ ]:





# In[ ]:





# In[ ]:




