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
df = pd.read_csv('nba_logreg.csv')
df.iloc[:2]
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
class_counts = df['TARGET_5Yrs'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
plt.title('Distribution of TARGET_5Yrs classes')
plt.show()
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

columns_for_correlation = df[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV']]
for column in columns_for_correlation:
    if column != 'TARGET_5Yrs':
        correlation = df['TARGET_5Yrs'].corr(df[column])
        print(f"Correlation between {column} and TARGET_5Yrs: {correlation:.3f}")

plt.figure(figsize=(12,9))
ax = sns.heatmap(df[['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','TARGET_5Yrs']].corr(), annot=True, cmap='RdYlBu')
ax.set_title("Correlation between Player Attributes", fontsize=20)
sns.despine()

column1 = df[['GP', 'MIN', 'PTS']]
sns.pairplot(column1, diag_kind='scatter')
plt.show()


null_counts = df.isnull().sum()

for column, null_count in null_counts.items():
    if null_count > 0:
        print(f"Column '{column}' has {null_count} null value(s)")

df.info()


#skewness = df.skew()
#print(skewness)

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
#cleaned_df_skewness = cleaned_df.skew()
#print(cleaned_df_skewness)


# In[21]:


print('Original data set shape',df.shape)
print('Without outlier data set shape',cleaned_df.shape)


# <h4> Imputing Null values with mean value</h4>

# In[22]:

cleaned_df = cleaned_df.drop('Name',axis=1)
cleaned_df_imputed = cleaned_df.fillna(cleaned_df.mean())

null_counts_new = cleaned_df_imputed.isnull().sum()

for column, null_count in null_counts_new.items():
    if null_count >= 0:
        print(f"Column '{column}' has {null_count} null value(s)")


# <h4> Class Balancing using SMOTE </h4>

# In[23]:


from imblearn.over_sampling import SMOTE
import pandas as pd

X = cleaned_df_imputed.drop(['TARGET_5Yrs'], axis=1)
y = cleaned_df_imputed['TARGET_5Yrs']

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new DataFrame with the resampled data
resampled_df = pd.concat([X_resampled, y_resampled], axis=1)
resampled_df.shape


class_counts = resampled_df['TARGET_5Yrs'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightyellow', 'skyblue'])
plt.title('Distribution of TARGET_5Yrs after SMOTE')
plt.show()


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
#rookies.drop('Name',axis=1,inplace=True)

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

X = rookies.drop(['TARGET_5Yrs','index'],axis=1)
y = rookies.loc[:,'TARGET_5Yrs']
X.shape


# In[31]:


#Case 1 Train = 80 % Test = 20% [ x_train1, y_train1] = 80%;[ x_test1, y_test1] = 20%;
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
X_train.shape



#Case 2: Train = 10 % Test = 90% [ x_train2, y_train2] = 10%;[ x_test2, y_test2] = 90% 
X_train2 , X_test2, y_train2, y_test2 = train_test_split(X,y,test_size = 0.9)
X_train2.shape


# Code to use K fold for cross validation
from sklearn.model_selection import KFold, cross_val_score
num_folds = 6
kf = KFold(n_splits=num_folds,shuffle=True,random_state=42)
# K-fold cross validation

#Model Deployment
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty=None,random_state=42)
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


dt_reg = DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,random_state=42)
dt_reg.fit(X_train,y_train)
#predict and calculate accuracuy
y_pred_dt_reg = dt_reg.predict(X_test)

print("Accuracy of Logistic regression without Regularization:"   , accuracy_score(y_test,y_lr))
print("Accuracy of Logistic regression with L2 Regularization:"   , accuracy_score(y_test,y_pred_lr_reg))
print("-------------------------------------------------------")
#Decision Tree
print("Accuracy of Decision Tree without Regularization:      " , accuracy_score(y_test,y_dt))
print("Accuracy of Decision Tree with Regularization:         "   , accuracy_score(y_test,y_pred_dt_reg))
print("Classification Report of Logistic Regression without Regularization and split 1 ")
print(f"{classification_report(y_test,y_lr)}")
print("Classification Report of Logistic Regression without Regularization and split 2 ")
print(f"{classification_report(y_test2,y_lr2)}")
print("Classification Report of Decision tree without Regularization and split 1 ")
print(f"{classification_report(y_test,y_dt)}")

print("Classification Report of Decision tree without Regularization and split 2 ")
print(f"{classification_report(y_test2,y_dt2)}")


train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    lr, X=X_train, y=y_train, train_sizes=train_sizes
)
plot.learning_curve(train_scores, test_scores, train_sizes)


train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    dt, X=X_train, y=y_train, train_sizes=train_sizes
)
plot.learning_curve(train_scores, test_scores, train_sizes)
pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))