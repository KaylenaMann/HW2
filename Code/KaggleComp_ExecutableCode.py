#!/usr/bin/env python
# coding: utf-8

# # HW2 Kaggle Competition: Stroke Prediction: End-to-End ML Pipeline

# ## Data loading and cleaning

# In[1]:


import pandas as pd
train_df = pd.read_csv("/Users/kaylenamann/Downloads/BC Grad/2025Fall_ADAN7430/assignments/HW2/RawData/train.csv")
print(train_df.head())
print(train_df.info()) 

# importing variables of choice
import statsmodels.api as sm
feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'Residence_type','avg_glucose_level', 'bmi', 'smoking_status']
X = pd.get_dummies(train_df[feature_cols], drop_first=True)
y = train_df['stroke']
X = sm.add_constant(X, has_constant='add')
X = X.astype(float)


# Data looks good, with no missing values or unexpected results

# ## Exploratory analysis 

# ### Checking the balance of our outcome

# In[2]:


stroke_counts = train_df['stroke'].value_counts()
print("Frequency Counts:")
print(stroke_counts)


# Our outcome is severely unbalanced, with far more non-strokes (0) than strokes (1), we should be mindful of this when tuning parameters and deciding on our model.

# ### Correlations

# In[3]:


corr_df = train_df[['avg_glucose_level', 'bmi', 'age', 'gender', 'hypertension', 'heart_disease', 'Residence_type', 'smoking_status']].copy()

from sklearn.preprocessing import LabelEncoder

le_gender = LabelEncoder()
le_residence = LabelEncoder()
le_smoking = LabelEncoder()

corr_df['gender'] = le_gender.fit_transform(corr_df['gender'])
corr_df['Residence_type'] = le_residence.fit_transform(corr_df['Residence_type'])
corr_df['smoking_status'] = le_smoking.fit_transform(corr_df['smoking_status'])

corr = corr_df.corr()
print(corr)


# Prior research has shown that smoking status is related to stroke indicators (Gan et al., 2018; Zhou et al., 2008), and can often interact. These correlations above add evidence to support this, with smoking having a moderate relationship with age and BMI. Additionally, BMI and age have a moderate relationship, so these interactions will be added to the model.

# ### Centering metric features and creating interactions 

# In[4]:


from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# doing a train/test split (ON THE TRAINING DATA ONLY)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39, stratify=y)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# ###  Running Diagnostic checks for non-linearity - Box-Tidwell
# 
# sm.GLM was used here to conduct the Box-Tidwell test in order to see the actual coefficients and p-values. None of the log-transformed interaction terms were significant (*p* > .05), indicating that the assumption of linearity in the logit was likely met.

# In[5]:


import matplotlib.pyplot as plt
import numpy as np

continuous_vars = ['age', 'avg_glucose_level', 'bmi']

# Running an original model without the log variables
model_original = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results_original = model_original.fit()
print(results_original.summary())

# Creating log interaction versions of the original variables
X_train_bt = X_train.copy()
for var in continuous_vars:
    X_train_bt[f'{var}_logint'] = X_train_bt[var] * np.log(X_train_bt[var] + 1)
    
model_bt = sm.GLM(y_train, X_train_bt, family=sm.families.Binomial())
results_bt = model_bt.fit()

# Comparing AIC and BIC between models
print("Original AIC:", results_original.aic)
print("Box–Tidwell AIC:", results_bt.aic)
print("\nBox–Tidwell model summary:")
print(results_bt.summary())


# ## Feature Engineering
# LogisticRegression for sklearn was used for the actual parameter tuning, since we can more easily add weights to account for imbalanced outcome. 

# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import get_scorer_names
from sklearn.preprocessing import StandardScaler

# Centering metric variables
for col in ["age", "bmi", "avg_glucose_level"]:
    mean_c = X_train[col].mean()
    X_train[f"{col}_c"] = X_train[col] - mean_c
    X_test[f"{col}_c"] = X_test[col] - mean_c


# In[7]:


# Dropping the constant because LogisticRegression code does not need it
X_train_for_sklearn = X_train.drop(columns=['const', 'age', 'bmi', 'avg_glucose_level'])
X_test_for_sklearn = X_test.drop(columns=['const', 'age', 'bmi', 'avg_glucose_level'])

# Adding in interaction terms to our model
X_train_for_sklearn["age_bmi_int"] = X_train_for_sklearn["age_c"] * X_train_for_sklearn["bmi_c"]
X_test_for_sklearn["age_bmi_int"] = X_test_for_sklearn["age_c"] * X_test_for_sklearn["bmi_c"]

for col in X_train_for_sklearn.columns:
    if 'smoking_status' in col:  
        X_train_for_sklearn[f'age_{col}_int'] = X_train_for_sklearn['age_c'] * X_train_for_sklearn[col]
        X_train_for_sklearn[f'bmi_{col}_int'] = X_train_for_sklearn['bmi_c'] * X_train_for_sklearn[col]
        X_test_for_sklearn[f'age_{col}_int'] = X_test_for_sklearn['age_c'] * X_test_for_sklearn[col]
        X_test_for_sklearn[f'bmi_{col}_int'] = X_test_for_sklearn['bmi_c'] * X_test_for_sklearn[col]


# In[8]:


# Testing different thresholds for our logistic regression
cols_to_scale = [c for c in ["age_c", "bmi_c", "age_bmi_int", "avg_glucose_level_c"]
                 if c in X_train_for_sklearn.columns]

scaler = StandardScaler()
X_train_for_sklearn[cols_to_scale] = scaler.fit_transform(X_train_for_sklearn[cols_to_scale])
X_test_for_sklearn[cols_to_scale]  = scaler.transform(X_test_for_sklearn[cols_to_scale])

scorer = make_scorer(f1_score, pos_label=1)
base_model = LogisticRegression(max_iter=2000, class_weight="balanced")
model = TunedThresholdClassifierCV(base_model, scoring=scorer, cv=5)
model.fit(X_train_for_sklearn, y_train)

#Double-checking that no variables were written-over
print(X_test_for_sklearn.isnull().sum())
print(np.isinf(X_test_for_sklearn).sum())
print(X_test_for_sklearn.columns)
print(X_train_for_sklearn.columns)


# In[39]:


print(model.best_threshold_)
print(model.best_score_)
print("Training features:", X_train_for_sklearn.shape)
print("Test features:", X_test_for_sklearn.shape)
print("Training columns:", X_train_for_sklearn.columns.tolist())
print("Test columns:", X_test_for_sklearn.columns.tolist())
print("Any NaN/Inf?", X_train_for_sklearn.isnull().sum().sum(), np.isinf(X_train.values).sum())


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

#Visually plotting our threshold values against F1-scores

probs = model.estimator_.predict_proba(X_test_for_sklearn)[:, 1]
thresholds = np.linspace(0.05, 0.95, 50)
f1s = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]

plt.plot(thresholds, f1s)
plt.axvline(model.best_threshold_, color='red', linestyle='--', label='Best threshold')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold")
plt.legend()
plt.show()


# ## Model Selection on the holdout test set
# 
# Now we will generate predictions on our holdout validation set **(remember this is held out from the training set, not the actual competition test set)** using different models. 

# ### Binary Logistic Regression

# In[11]:


from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, recall_score, precision_score,
    precision_recall_curve, auc
)

y_pred_logistic = model.predict(X_test_for_sklearn)


# ### KNN

# In[12]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer

f1_results = {}

#Testing different kneighbors
for k in [5, 7, 9, 11, 15, 17, 21, 23, 25]:
    base_model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    model = TunedThresholdClassifierCV(base_model, scoring=scorer, cv=5)
    model.fit(X_train_for_sklearn, y_train)
    y_pred = model.predict(X_test_for_sklearn)
    f1 = f1_score(y_test, y_pred)
    print(f"KNN k={k}: F1 = {f1:.4f}, Threshold = {model.best_threshold_:.3f}")


# ### Binary Logistic Regularization (L2) model

# In[13]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_recall_curve, auc

logistic_l2 = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', 
                                  class_weight='balanced', random_state=39)
logistic_l2.fit(X_train_for_sklearn, y_train)  

y_pred_l2 = logistic_l2.predict(X_test_for_sklearn)


# ## Model evaluation and comparison
# 
# Finally, we compare evaluation metrics such as the confusion matrix and classifcation report along with Accuracy and ROC curve. We focus mainly of the F1-score and the precision recall curve since we have an imbalanced outcome. 

# ### Regular Logistic Regression Results

# In[16]:


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic))
print("Classification Report:\n", classification_report(y_test, y_pred_logistic, digits=3))
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("ROC AUC:", roc_auc_score(y_test, probs))

prec, rec, thr = precision_recall_curve(y_test, probs)
pr_auc = auc(rec, prec)
print("PR-AUC:", pr_auc)


# ### KNN Results

# In[17]:


best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=21))
])
final_model = TunedThresholdClassifierCV(best_model, scoring=scorer, cv=5)
final_model.fit(X_train_for_sklearn, y_train)

y_pred_knn = final_model.predict(X_test_for_sklearn)
probs_knn  = final_model.estimator_.predict_proba(X_test_for_sklearn)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn, digits=3))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("ROC AUC:", roc_auc_score(y_test, probs_knn))

prec, rec, thr = precision_recall_curve(y_test, probs_knn)
pr_auc = auc(rec, prec)
print("PR-AUC:", pr_auc)


# ### L2 Regularization Results

# In[18]:


probs = logistic_l2.predict_proba(X_test_for_sklearn)[:, 1]

accuracy_l2 = accuracy_score(y_test, y_pred_l2)
print(f"Accuracy with L2 regularization: {accuracy_l2:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_l2))
print("Classification Report:\n", classification_report(y_test, y_pred_l2, digits=3))
print("ROC AUC:", roc_auc_score(y_test, probs))

prec, rec, thr = precision_recall_curve(y_test, probs)
pr_auc = auc(rec, prec)
print("PR-AUC:", pr_auc)


# ## Using our final model on the test set

# In[28]:


test_df = pd.read_csv("/Users/kaylenamann/Downloads/BC Grad/2025Fall_ADAN7430/assignments/HW2/RawData/test.csv")

print("Test data shape:", test_df.shape)
print(test_df.head())


# In[29]:


feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
X_test_kaggle = test_df[feature_cols].copy()

X_test_kaggle = pd.get_dummies(X_test_kaggle, drop_first=True)
X_test_kaggle = X_test_kaggle.astype(float)
X_test_kaggle = X_test_kaggle.reindex(columns=X_train.columns, fill_value=0)


# In[30]:


#repeating the same process
for col in ["age", "bmi", "avg_glucose_level"]:
    mean_c = X_train[col].mean() 
    X_test_kaggle[f"{col}_c"] = X_test_kaggle[col] - mean_c

X_test_kaggle["age_bmi_int"] = X_test_kaggle["age_c"] * X_test_kaggle["bmi_c"]

for col in X_test_kaggle.columns:
    if 'smoking_status_' in col:
        X_test_kaggle[f'age_{col}_int'] = X_test_kaggle['age_c'] * X_test_kaggle[col]
        X_test_kaggle[f'bmi_{col}_int'] = X_test_kaggle['bmi_c'] * X_test_kaggle[col]

X_test_kaggle = X_test_kaggle.drop(columns=['const','age', 'bmi', 'avg_glucose_level'])


# In[31]:


cols_to_scale = [c for c in ["age_c", "bmi_c", "age_bmi_int", "avg_glucose_level_c"]
                 if c in X_test_kaggle.columns]

X_test_kaggle[cols_to_scale] = scaler.transform(X_test_kaggle[cols_to_scale])

#Checking that our test columns match our training columns
print("Test features shape:", X_test_kaggle.shape)
print("Test columns:", X_test_kaggle.columns.tolist())


# In[32]:


y_pred_final = model.predict(X_test_kaggle)

#making sure our shape is correct and the number of predictions
print("Predictions shape:", y_pred_final.shape)
print("Number of predicted strokes:", (y_pred_final == 1).sum())
print("Number of predicted no strokes:", (y_pred_final == 0).sum())


# In[37]:


#creating the final csv file for submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'stroke': y_pred_final
})

print("\nSubmission preview:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")
print(f"Submission columns: {submission.columns.tolist()}")

submission.to_csv('/Users/kaylenamann/Downloads/BC Grad/2025Fall_ADAN7430/assignments/HW2/Output/Submission.csv', index=False)
print("\nSubmission saved to Output/submission.csv")


# ## Conclusion

# | Model | F1   | PR-AUC |
# |------:|:----:|-------:|
# | LogReg| 0.33 | 0.25   |
# | KNN   | 0.31 | 0.20   |
# | L2Reg | 0.24 | 0.25   |

# ### The features and outcome were first evaluated descriptively to check for imbalance, non-linearity, and potential interactions.
# 
# - The outcome was heavily imbalanced with very few cases of stroke (1).
# - There were a few moderate correlations between variables, indicating potential for interactions (in line with research).
#     - BMI and Age
#     - Age and smoking status
#     - BMI and smoking status
# - No evidence of non-linearity with the logit for the continuous variables, so no polynomial terms were added.
# 
# ### Different models were tested with the goal of maximizing the F1-score to catch more cases of stroke
# 
# - Logistic regression with class weights added and tuned threshold
#     - F1-Score: .33
# - KNN with class weights added and tuned threshold
#     - F1-Score: .32
# - L2 regularization with class weights
#     - F1-Score: .24
# 
# ### **Overall, the normal logistic regression performed the best on the holdout set with an F1-score of .33, when submitted to the Kaggle competition, our final score was .28**

# References: 
# 
# Gan, Y., Wu, J., Li, L., Zhang, S., Yang, T., Tan, S., Mkandawire, N., Zhong, Y., Jiang, J., Wang,Z., & Lu, Z. (2018). Association of smoking with risk of stroke in middle-aged and older Chinese: Evidence from the China National Stroke Prevention Project. *Medicine, 97*(47), e13260. https://doi.org/10.1097/MD.0000000000013260
# 
# Zhou, L., Chen, G., Liu, C., Liu, L., Zhang, S., & Zhao, X. (2008). Body mass index, blood pressure, and mortality from stroke: a prospective cohort study. *Stroke, 39*(7), 2065–2071. https://doi.org/10.1161/STROKEAHA.107.495374
