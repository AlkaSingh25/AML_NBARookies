#!/usr/bin/env python
# coding: utf-8

# ![BITS_university_logo.gif](attachment:BITS_university_logo.gif)

# <h1><center>Work Integrated Learning Programmes Division<br>
# M.Tech (Data Science and Engineering)<br> Applied Machine Learning (DSECL ZG568))<br>
# Second Semester, 2021-22
# </center></h1>
# 
# <h2><center>Assignment – Problem Statement 4 <br>
# NBA Rookies [Weightage 30%] </center></h2>

# <h3>Instructions for Assignment Evaluation:</h3>

# <ol>
# <li>Please follow the naming convention as Group no_Dataset name.ipynb.</li>
# Eg – for group 1 with a weather dataset your notebooks should be named as - Group1_NBA Rookies.ipynb. 
# <li>Inside each jupyter notebook, you are required to mention your name, Group details and the Assignment dataset you will be working on.</li>
# <li>Organize your code in separate sections for each task. Add comments to make the code readable.</li>
# <li>Deep Learning Models are strictly not allowed. You are encouraged to learn classical Machine learning techniques and experience their behavior. For comparison of output with classical model you can use, if needed.</li> 
# <li>Notebooks without output shall not be considered for evaluation.</li>
# <li>Delete unnecessary error messages and long outputs.</li>
# <li>Display the analysis of attributes in one frame rather than one after one. However, special treatment to attributes can be displayed separately.</li>
# <li>Prepare a jupyter notebook (recommended - Google Colab) to build, train and evaluate a Machine Learning model on the given dataset. Please read the instructions carefully.</li>
# <li>Each group consists of up to 4 members. All members of the group will work on the same problem statement.</li>
# <li>Only two files should be uploaded in canvas without zipping them.  One is ipynb file and other one html output of the ipynb file.  No other files should be uploaded.</li>
# <li>Each group should upload in CANVAS in respective locations under ASSIGNMENT Tab. Assignment submitted via means other than through CANVAS will not be graded.</li>
#     </ol>
# 

# <style>
# table {
#   font-family: arial, sans-serif;
#   border-collapse: collapse;
#   width: 100%;
# }
# 
# td, th {
#   border: 1px solid #dddddd;
#   text-align: left;
#   padding: 8px;
# }
# 
# tr:nth-child(even) {
#   background-color: #dddddd;
# }
# </style>
# 
# <h2>Group No: 41</h2>
# <h3>Dataset: NBA Rookie</h3>
# 
# <table>
#   <tr>
#     <th>S. No</th>
#     <th>Team member name</th>
#     <th>BITS ID</th>
#   </tr>
#   <tr>
#       <th>1.</th>
#       <td>Ajayveer Singh</td>
#       <td>2022DC04140</th>
#   <tr>
#     <th>2.</th>
#     <td>Alka Singh</td>
#     <td>2022DC04028</td>
#   </tr>
#   <tr>
#     <th>3.</th>
#     <td>Shivangi Shukla</td>
#     <td>2022DC04193</td>
#   </tr>
#     <tr>
#     <th>4.</th>
#     <td>Wanwe Apurva Shyam</td>
#     <td>2022DC04499</td>
# </table>

# <h3>Problem Statement:</h3>
# <h4>Classification Exercise: Predict 5-Year Career Longevity for NBA Rookies y = 0 if career years played < 5 y = 1 if career years played >= 5</h4>

# ## Downloading dataset and Importing Libraries

# <h4>Import the required libraries</h4>

# In[1]:


#pip install imbalanced-learn


# In[2]:


#pip install -U threadpoolctl


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy import stats
import numpy as np

warnings.filterwarnings('ignore')


### Regularization
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report
from sklearn.model_selection import learning_curve
from sklearn_evaluation import plot, table
import pickle


# <h4>Loading datatset and storing in dataframe</h4>

# In[4]:


df = pd.read_csv('nba_logreg.csv')


# ## Data Visualization and Exploration [4 M]

# <h4>1.1 Print 2 rows for sanity check</h4>

# In[5]:


df.style.hide_index()
df.iloc[:2]


# <h4>1.2 Class Imbalance Visualization</h4>

# In[6]:


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
class_counts = df['TARGET_5Yrs'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
plt.title('Distribution of TARGET_5Yrs classes')
plt.show()


# <h4>1.3 Comments on above visualization</h4>
# 
# It is evident from the above Pie Chart visulaization, that there is clear class imablance in the Target column where category 1 is in majority (62.0% of total data set) and 0 category is minority category (38.0%).

# <h4>1.4 Visuals to be used to get insight on data </h4>

# To get the insight on the data we can plot the Histograms as it will help us to better visualize the underlying distribution of the independent variable, we ill plot Histogram in goups of:
# 1. Games Played, Points Per Game, Minutes Played
# 2. Field Goals
# 3. Three Pointers
# 4. Free Throws
# 5. Rebounds
# 6. Assist, Steals, Blocks, Turnovers

# <h4>a) Games Played, Points Per Game, Minutes Played </h4>

# In[7]:


plt.figure(figsize=(16,6))
plt.subplot(1, 3, 1)
data = df['GP']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Net Income')
plt.ylabel('GP')

plt.subplot(1, 3, 2)
data = df['MIN']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Net Income')
plt.ylabel('MIN')

plt.subplot(1, 3, 3)
data = df['PTS']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Net Income')
plt.ylabel('PTS')


# <h4> Observations </h4>
# 
# The Minutes Played and Points Per Game variables appear to exhibit a right-skewed distribution, while the Games Played variable appears to be mostly left-skewed.

# <h4>b) Field Goals</h4>

# In[8]:


plt.figure(figsize=(16,6))
plt.subplot(1, 3, 1)
data = df['FGM']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Field Goals Made')
plt.ylabel('FGM')

plt.subplot(1, 3, 2)
data = df['FGA']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Field Goals Attempted')
plt.ylabel('FGA')

plt.subplot(1, 3, 3)
data = df['FG%']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Field Goals Percent')
plt.ylabel('FG%')


# <h4> Observations </h4>
# 
# The Field Goals Made and Field Goals Attempted variables appear to exhibit a right-skewed distribution, while the Field Goals Percentage variable appears to mostly resemble a normal distribution.

# <h4>c) Three Points</h4>

# In[9]:


plt.figure(figsize=(16,6))
plt.subplot(1, 3, 1)
data = df['3P Made']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - 3 Point Made')
plt.ylabel('3P Made')

plt.subplot(1, 3, 2)
data = df['3PA']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - 3 Point Attemptes')
plt.ylabel('3PA')

plt.subplot(1, 3, 3)
data = df['3P%']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - 3 Point Attemptes %')
plt.ylabel('3P%')


# <h4> Observations </h4>
# 
# he Three Pointers Made and Three Pointers Attempted variables both exhibit a positively skewed distribution, while the Three Pointer Percentage variable appears to be bimodal.

# <h4>d) Free Throws </h4>

# In[10]:


plt.figure(figsize=(16,6))
plt.subplot(1, 3, 1)
data = df['FTM']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Free Throws Made')
plt.ylabel('FTM')

plt.subplot(1, 3, 2)
data = df['FTA']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Free Throws Attempted')
plt.ylabel('FTA')

plt.subplot(1, 3, 3)
data = df['FT%']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Free Throw Percent')
plt.ylabel('FT%')


# <h4> Observations </h4>
# 
# The distributions of Free Throws Made and Free Throws Attempted both appear to be positively skewed, while the distribution of Free Throw Percentage mostly resembles a negatively skewed distribution.

# <h4>e) Rebounds </h4>

# In[11]:


plt.figure(figsize=(16,6))
plt.subplot(1, 3, 1)
data = df['OREB']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Offensive Rebounds')
plt.ylabel('OREB')

plt.subplot(1, 3, 2)
data = df['DREB']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Defensive Rebounds')
plt.ylabel('DREB')

plt.subplot(1, 3, 3)
data = df['REB']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Rebounds')
plt.ylabel('REB')


# <h4> Observations </h4>
# 
# The distributions of offensive rebounds, defensive rebounds, and total rebounds all appear to be mostly positively skewed.

# <h4>f) Assist, Steals, Blocks, Turnovers </h4>

# In[12]:


plt.figure(figsize=(16,6))
plt.subplot(1, 4, 1)
data = df['AST']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Assists')
plt.ylabel('AST')

plt.subplot(1, 4, 2)
data = df['STL']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Steals')
plt.ylabel('STL')

plt.subplot(1, 4, 3)
data = df['BLK']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Blocks')
plt.ylabel('BLK')

plt.subplot(1, 4, 4)
data = df['TOV']
plt.hist(data)
plt.ticklabel_format(style='plain', axis='y')  # Display large integers as they are
plt.title('Hist Plot - Turnover')
plt.ylabel('TOV')


# <h4> Observations </h4>
# 
# The distributions of assists, steals, blocks, and turnovers all appear to be mostly positively skewed.

# <h4>1.5 Correlation Analysis </h4>

# In[13]:


columns_for_correlation = df[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV']]
for column in columns_for_correlation:
    if column != 'TARGET_5Yrs':
        correlation = df['TARGET_5Yrs'].corr(df[column])
        print(f"Correlation between {column} and TARGET_5Yrs: {correlation:.3f}")


# In[14]:


plt.figure(figsize=(12,9))
ax = sns.heatmap(df[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','TARGET_5Yrs']].corr(), annot=True, cmap='RdYlBu')
ax.set_title("Correlation between Player Attributes", fontsize=20)
sns.despine()


# <h4> Impact on feature selection</h4>
# 
# i) Correlation analysis was performed to explore the relationships between the variables of interest and the dependent variable, target5yrs. These matrices will play a crucial role in identifying the variables that best predict whether an NBA player will remain in the league for at least 5 years.
# 
# ii) It is evident that GP (Games Played), MIN (Minutes Played), PTS (Points per game) are having highest correlation with the traget variable
# 
# iii) It's important to consider that variables with a correlation above 0.3 or below -0.3 are valuable for predicting a player's draft status. A correlation of 0.3 signifies a moderate positive relationship, while a correlation of -0.3 indicates a moderate negative relationship.

# <h4> 1.6 Any other Visualization</h4>

# In[15]:


column1 = df[['GP', 'MIN', 'PTS']]
sns.pairplot(column1, diag_kind='scatter')
plt.show()


# <h4> Scatter Plot Matrix </h4>
# 
# This can help visualize the strength and direction of relationships between features. For exploarion of other visulas pairplot is done between GP, MIN, PTS features.

# ## 2. Data Pre-processing and cleaning [4 M]

# <h3>2.1 Data Pre-processing </h3>

# For data pre-processing we have done following two process:
# 1. Identified and Imputed Null values from the dataset
# 2. Identified and removed outliers from the dataset

# <h4>2.1.1 Identifying Null values</h4>

# In[16]:


null_counts = df.isnull().sum()

for column, null_count in null_counts.items():
    if null_count > 0:
        print(f"Column '{column}' has {null_count} null value(s)")


# <h4> Imputing Null values </h4>

# For imputing the null values we would first like to know the data type as it will play an important role in selecting imputing technique.

# In[17]:


# Attribute type of each column
df.info()


# <h4> Observation </h4>
# 
# As the columns with null values are of float type, so we can either use Mean or Median Imputation technique, but before doing that we should check the skewness of the data as outlier may influence the mean or median value.

# In[18]:


skewness = df.skew()
print(skewness)


# <h4> Observation </h4>
# 
# As data is skewed we should first identify the outliers and remove it from the dataset before null imputation

# <h4> Outlier Detection </h4>

# In[19]:


# calcualte z-scores of numeric attributes
z_scores = stats.zscore(columns_for_correlation)
# Create a DataFrame from z-scores
z_score_df = pd.DataFrame(z_scores, columns=df.columns)

# Identify outliers using a threshold (e.g., z-score > 3 or < -3)
outliers = z_score_df[(z_score_df > 3) | (z_score_df < -3)]

# Print the rows containing outliers
outliers_count = outliers.count()

# Print the count of outliers in each column
print("Count of outliers in each column:")
print(outliers_count)


# <h4>Outlier Removal</h4>

# In[20]:


# removing outliers
cleaned_df = df[~outliers.any(axis=1)]
cleaned_df_skewness = cleaned_df.skew()
print(cleaned_df_skewness)


# In[21]:


print('Original data set shape',df.shape)
print('Without outlier data set shape',cleaned_df.shape)


# <h4> Imputing Null values with mean value</h4>

# In[22]:


cleaned_df_imputed = cleaned_df.fillna(cleaned_df.mean())

null_counts_new = cleaned_df_imputed.isnull().sum()

for column, null_count in null_counts_new.items():
    if null_count >= 0:
        print(f"Column '{column}' has {null_count} null value(s)")


# <h4> Class Balancing using SMOTE </h4>

# In[23]:


from imblearn.over_sampling import SMOTE
import pandas as pd

X = cleaned_df_imputed.drop(['TARGET_5Yrs', 'Name'], axis=1)
y = cleaned_df_imputed['TARGET_5Yrs']

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new DataFrame with the resampled data
resampled_df = pd.concat([X_resampled, y_resampled], axis=1)
resampled_df.shape


# In[ ]:





# <h4> Visualizing the output of SMOTE balancing</h4>

# In[24]:


class_counts = resampled_df['TARGET_5Yrs'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightyellow', 'skyblue'])
plt.title('Distribution of TARGET_5Yrs after SMOTE')
plt.show()


# <h4> Explanation: </h4>
# 
# In this code, SMOTE (Synthetic Minority Over-sampling Technique) is used to create synthetic samples 
# for the minority class (class 0) to balance the classes 
# We also see an increase in the no of rows.
# 
# <h4> NOTE: </h4> 
# 
# For the Modeling purpose we have only explored SMOTE as one of the technique to balance the class. We haven't used balanced data in final analysis, because SMOTE syntehctically adds data hence, this may give us problems like overfitting of model and loss of information in later stages.

# <h3> 2.1 Feature Engineering & Feature Transformation </h3>

# <h4> Standardization

# As most of the data is not normalized which was evident from the histograms exhibited in EDA part of the analysis. So we would be utilizing the Standardization technique to centralize the data and follow a normal distribution.

# In[25]:


cleaned_df_imputed.info()


# In[26]:


from scipy.stats import shapiro

# Selecting only the numerical columns for normality test
numeric_columns = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']

# Creating a new DataFrame with selected columns
selected_df = cleaned_df_imputed[numeric_columns]

# Perform the Shapiro-Wilk test for each column
normality_results = {}
for column in selected_df.columns:
    _, p_value = shapiro(selected_df[column])
    normality_results[column] = p_value

# Display the p-values
normality_results_df = pd.DataFrame.from_dict(normality_results, orient='index', columns=['p-value'])
print(normality_results_df)


# <h4> Observation: </h4>
# 
# As we can see the numerical columns are having p-value less than 0.05 hance data is not having normal distribution

# <h4> Data Standardization </h4>

# In[27]:



scaler = StandardScaler()

# Fit and transform the selected columns
scaled_features = scaler.fit_transform(selected_df)

# Creating a new DataFrame with scaled features
scaled_df = pd.DataFrame(scaled_features, columns=numeric_columns)

# Print the first few rows of the scaled DataFrame
print(scaled_df.head())


# In[28]:


numeric_columns = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']
rookies = pd.concat([scaled_df,cleaned_df_imputed.drop(numeric_columns, axis=1).reset_index()],axis=1)
rookies.drop('Name',axis=1,inplace=True)
#dropping nominal column "Name"


# <h4> Feature Selection </h4>
# 
# We have used Anova F Statistic to evaluate feature selection. The importannt features as seen from the graph are : GP, FGM, PTS

# In[29]:


from sklearn.feature_selection import f_classif
import pandas as pd
 
# Assuming merged_df is your DataFrame containing the data
ANOVA_X = rookies.drop('TARGET_5Yrs', axis=1)
ANOVA_y = rookies['TARGET_5Yrs']
 
# Calculate ANOVA F-statistic
f_scores, p_values = f_classif(ANOVA_X, ANOVA_y)
 
# Create a DataFrame to display the results
anova_results = pd.DataFrame({'Feature': ANOVA_X.columns, 'F-Score': f_scores,'p-value': p_values})
anova_results.sort_values('F-Score', ascending=False, inplace=True)
plt.figure(figsize=(12, 6))
plt.barh(anova_results['Feature'], anova_results['F-Score'])
plt.xlabel('F-Score')
plt.ylabel('Feature')
plt.title('ANOVA F-Scores for Numeric Features')
plt.gca().invert_yaxis()  # Invert y-axis to display features from top to bottom
plt.show()


# In[ ]:





# <h4> 3 Model Building </h4>

# 3.1 Train and Test Split of the dataset
# Explanation: If all the dataset is used in training the model, and then the model is applied to real-world, we risk overfitting. We can not evaluate the model also. The best practice here is to split the given dataset into train and test such that model can be trained on the train dataset and evaluated based on test dataset.
# 

# In[30]:


X = rookies.drop(['TARGET_5Yrs','index'],axis=1)
y = rookies.loc[:,'TARGET_5Yrs']
X.shape


# In[31]:


#Case 1 Train = 80 % Test = 20% [ x_train1, y_train1] = 80%;[ x_test1, y_test1] = 20%;
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
X_train.shape


# <h4> Note </h4>
# 
# We will use Case 1 split for all model comparisions

# In[32]:


#Case 2: Train = 10 % Test = 90% [ x_train2, y_train2] = 10%;[ x_test2, y_test2] = 90% 
X_train2 , X_test2, y_train2, y_test2 = train_test_split(X,y,test_size = 0.9)
X_train2.shape


# <h4> 3.2 Cross Validation: </h4>
# 
# Main purpose of cross validation is to prevent over-fitting. Overfitting occurs when the model performs well on the training data and poorly on test. There are many ways to prevent the overfitting and this is done by cross-validation. There are many ways of cross-validation like: Holdout Validation, Leave One out Cross Validation, Stratified, and k-fold validation.
# This is mainly used for fining hyper parameters like we will use Lambda in Regularization using cross validation.
# 
# <h4> K-fold validation: </h4>
# we split the dataset into k number of subsets (known as folds) then we perform training on the all the subsets but leave one(k-1) subset for the evaluation of the trained model. In this method, we iterate k times with a different subset reserved for testing purpose each time.

# In[33]:


# Code to use K fold for cross validation
from sklearn.model_selection import KFold, cross_val_score
num_folds = 6
kf = KFold(n_splits=num_folds,shuffle=True,random_state=42)
# K-fold cross validation


# 
# Below code can be used to perform k fold cross validation in order to tune hyperparameters <br>
# cross_val_results = cross_val_score(lr,X,y,cv=kf) <br>
# print(cross_val_results.mean()) <br>
# cross_val_results_dt = cross_val_score(dt,X,y,cv=kf) <br>
# print(cross_val_results_dt.mean()) <br>
# 

# <h4> Note </h4>
# As the above code needs the model, we will continue the K fold code laer 

# <h4> 3.2 Model Selection  </h4>
# The requirement is to predict a binary classification whether a rookie player continue to play after 5 years or drop out?
# There are many ways of impleenting in ML - Logictic Regression, Decision Tree , Random Forest etc. We are choosing Logistic Regression as this model predicts 0 and 1 , and Decision Tree as it can classify the players in 1 and 1 category.

# In[34]:


#Model Deployment
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='none',random_state=42)
#lr2 = LogisticRegression(penalty='none')

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
#dt2 = DecisionTreeClassifier()


# In[35]:


# Logistic Regression
#Train the models using Case 1 split
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)

#Train the models using Case 2 split
lr.fit(X_train2,y_train2)
y_lr2 = lr.predict(X_test2)


# In[36]:


#Decision Tree with Case 1 Split
dt.fit(X_train,y_train)
y_dt = dt.predict(X_test)
#Decision Tree with Split 2
dt.fit(X_train2,y_train2)
y_dt2 = dt.predict(X_test2)


# <h4>  Need of regularization </h4>
# Regularization is used to solve overfitting by penalizing the cost function.It does so by using a penalty term in the cost function.
# 
# <h4> Regularization for Logistic Regression </h4>
# The two wo types of Regulraization L1(Lasso) and L2(Ridge) commonly used for Logistic Regression. Lambda is the Regularization parameter.

# In[37]:


#Ridge L2 Regression
cross_val_scores_ridge = []
alpha = []
for i in range(1, 10):
    ridgeModel = Ridge(alpha = i * 0.25)
    ridgeModel.fit(X_train, y_train)
    scores = cross_val_score(ridgeModel,X , y, cv = 10)
    avg_cross_val_score = np.mean(scores)*100
    cross_val_scores_ridge.append(avg_cross_val_score)
    alpha.append(i * 0.25)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
   print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))


# In[38]:


# As alpha = 2.25 has highest cross_val_Score, we will use alpha = 2.25
ridgeModelChosen = Ridge(alpha=2.25)
ridgeModelChosen.fit(X_train,y_train)
y_pred_lr_reg = ridgeModelChosen.predict(X_test)
i=0
for k in y_pred_lr_reg:
    if  y_pred_lr_reg[i] < 0.5:
        y_pred_lr_reg[i] = 0
    else:
        y_pred_lr_reg[i]  = 1
    i = i +1     


# <h4> Note </h4>
# We have explored Regularization as pert of part of the assignment but we will not be using Regularization in the model prediction. Before applying regularization, we should know if the model is overfitting ot not. 
# 

# <h4> Regularization for Decision Tree </h4>
# The Decision Tree are prone to overfitting when they are deep.
# This occurs when model captures the noise fluctuations in training data and does not model the underlying data pattern.
# A tree that is with many branches, is complex, and will perform poorly on test data. That is high variance and low bias. 
# Regularization in a Decision Tree is handled using many ways:<br>
# Maximum Depth: setting limit on how deep tree can grow <br>
# Pruning removing parts of trr that do not provide power in predicting target <br>
# Maximum features: setting limit on the number of features considered for splitting, we can add feature regularization
# In our model, we will use max_depth

# In[39]:


dt_reg = DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,random_state=42)
dt_reg.fit(X_train,y_train)
#predict and calculate accuracuy
y_pred_dt_reg = dt_reg.predict(X_test)
#print("Accuracy of Decision Tree with Regularization: " , accuracy_score(y_test,y_pred_dt_reg))


# <h4> Comparision of Logistic Regression and Decision Tree with and without Regularization </h4>

# In[40]:


#Logistic Regression
print("Accuracy of Logistic regression without Regularization:"   , accuracy_score(y_test,y_lr))
print("Accuracy of Logistic regression with L2 Regularization:"   , accuracy_score(y_test,y_pred_lr_reg))
print("-------------------------------------------------------")
#Decision Tree
print("Accuracy of Decision Tree without Regularization:      " , accuracy_score(y_test,y_dt))
print("Accuracy of Decision Tree with Regularization:         "   , accuracy_score(y_test,y_pred_dt_reg))


# <h4> Observations </h4>
# 
# From the accuracy results, we can see that the accuracy has improved in general for the two model implemented. The logistic Regression has an improved accuracy.The Logistic Regression is penalized and alpha is the hyperparameter
# The accuracy of a Decision Tree has improved with Regularization. This is expected as the tree can grow dense qith many deep branches and this is an overfitted tree. By limiting the no of branches and the number of leaves, we have reduced the variance and increased the bias

# <h4> Classification Report </h4>

# In[41]:


print("Classification Report of Logistic Regression without Regularization and split 1 ")
print(f"{classification_report(y_test,y_lr)}")


# In[42]:


print("Classification Report of Logistic Regression without Regularization and split 2 ")
print(f"{classification_report(y_test2,y_lr2)}")


# In[43]:


print("Classification Report of Decision tree without Regularization and split 1 ")
print(f"{classification_report(y_test,y_dt)}")


# In[44]:


print("Classification Report of Decision tree without Regularization and split 2 ")
print(f"{classification_report(y_test2,y_dt2)}")


# <h4> Observations </h4>
# 
# 1) The class target is imbalnced, so changing the train_test_split from 80:20 to 10:90 will impact the target class. Precision and Recall are sensitive to change in imbalance and hence we see change in Precision and Recall.
# 
# 2) As our target class is imbalanced, accuracy may not be the best evaluation metric.Precision and recall are better evaluation metrics.Precision measures the proportion of true positive predictions among all positive predictions. Recall (also known as sensitivity) measures the proportion of true positive predictions among all actual positive instances. These metrics are particularly useful when the cost of false positives or false negatives is different. Precision and recall can be combined into a single metric called the F1-score, which is the harmonic mean of precision and recall.
# 

# <h4> Determining if a model is Overfitting /Underfitting/Balanced </h4>
# 
#     

# In[45]:


#Learning Curve plotting for Logistic Regression

train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    lr, X=X_train, y=y_train, train_sizes=train_sizes
)
plot.learning_curve(train_scores, test_scores, train_sizes)

## As we see that the training score and validation score is approaching together, thelogistic regression model is balanced


# <h4> Observation </h4>
# 
# We can see that the Cross valiation score and Training Score is converging. The model is giving comparable results with training and test data. Hence, we can see that the logistic Regression model is balanced. 
# 
# Note: Learning curve is applied on the non regularized model as the goal here is to find if a model is overfitting or not.

# In[46]:


# Learning Curve for Decision Tree
train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    dt, X=X_train, y=y_train, train_sizes=train_sizes
)
plot.learning_curve(train_scores, test_scores, train_sizes)
## decision tree is an overfit model


# <h4> Observation </h4>
# 
# The Decision tree model is overfitting as the model is able to give accurate predictions for training set while poorly performing for the validation set. The decision tree will need regularization as discussed in the above notebook
# 
# Note: Learning curve is applied on the non regularized model as the goal here is to find if a model is overfitting or not.

# <h4> Conclusion </h4>
# 

# The recommended model is Logistic Regression with 66% accuracy and can be used in real world to predict the longevity of a rookie.

# In[ ]:
pickle.dump(lr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))




# In[ ]:




