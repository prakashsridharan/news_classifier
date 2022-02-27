# -*- coding: utf-8 -*-
"""Prakash-Project-71 - Phase-III & IV - 2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b0c9Jae3cLvzwp-i8a5M88rQa9wQAb5D

**Mounting GDrive**
"""

# Mounting the drive
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

"""**Setting up Project Directories**"""

# Setting folder path of the project/data files
data_path = 'drive/My Drive/Training resources/Dip in AL ML/project-71/'

"""**Importing Necessary Libraries**"""

# Importing necessary packages
# numpy, pandas for handling data
import numpy as np
import pandas as pd

# For handling data
import scipy

# For Plotting Charts - matplotlib, seaborn, plotly
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, BaggingRegressor, GradientBoostingClassifier,BaggingClassifier
from sklearn.naive_bayes import GaussianNB

# Tfidf and other packages
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, BaggingRegressor, GradientBoostingClassifier,BaggingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics.pairwise import linear_kernel

# Performance metrics
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn.model_selection import ShuffleSplit

#Saving the models into files using joblibs
import joblib

"""**Reading Analysed data from the file**


"""

# Reading the Data file (stored after data analysis and cleaning) in CSV format and storing the data into the Dataframe
df = pd.read_csv('drive/My Drive/Training resources/Dip in AL ML/project-71/news_datafame.csv')
df.info()

"""**Method to retrieve Category Name based on Category Id**"""

# get_catagory_name method returns the category name based on the Category_id.  This is used to show the classification report 
# based on Category name instead of category id

df_catagory_mapping=df.drop_duplicates(["clean_category_id", "clean_category"])[["clean_category_id", "clean_category"]]

def get_catagory_name(cat_id):
  return df_catagory_mapping.loc[df_catagory_mapping['clean_category_id'] == cat_id, 'clean_category'].iloc[0]

get_catagory_name(5)

"""**Error Analysis - Creating Features from errors**

Error Analysis helps to improve the performance of the model.  Errors can be converted as features.  We follow the below steps for generating features from error analysis

*   Step - 1: As this is a multi class classifier problem, the data is split into 2 categories.  Category - 1 represents the category with highest count of documents.  In this case, it is POLITICS.  Category - 0 represents other data
*   Step - 2:  Apply Logistic Regression for the entire dataset and find the High proba & low proba indexes for correct & wrong predictions and create features(Label 1 to 4) based on that
*   Step - 3:  Apply Logistic Regression again, find the error features and add these features to the original dataframe.
*   Step - 4:  Apply the model with the new error features and see if there is any improvement in the accuracy.  In this case, there is an increase of 3% in the accuracy



"""

###############################################################################
###########  ERROR ANALYSIS - FEATURES FROM ERRORS  ###########################
###############################################################################
## To get the features from Error.  Adding a new column in the Dataframe - binary_category
#  As this is multi class classification, splitting the data into 2 categories for error analysis
#  1 - The category that has more counts - POLITICS will be considered as 1
#  0 - Others will be considered as 0

df["binary_category"] = 0
df.loc[df['clean_category'] == 'POLITICS', 'binary_category'] = 1

print(df.binary_category.value_counts())

# Parameter selection
ngram_range = (1,3)
min_df = 10
max_df = 1.
max_features = 10000

# Forming feature list using clean_news_text, clean_link, and clean_authors data and applying the regression model
vectorizer_whole_news = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

vectorizer_whole_news.fit_transform(df["clean_news_text"].values.astype('U'))
X_whole_news_vect = vectorizer_whole_news.transform(df["clean_news_text"].values.astype('U'))

vectorizer_whole_link = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
vectorizer_whole_link.fit_transform(df["clean_link"].values.astype('U'))
X_whole_link_vect = vectorizer_whole_link.transform(df["clean_link"].values.astype('U'))

vectorizer_whole_authors = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
vectorizer_whole_authors.fit_transform(df["clean_authors"].values.astype('U'))
X_whole_authors_vect = vectorizer_whole_authors.transform(df["clean_authors"].values.astype('U'))

# Combining TFIDF features of clean_news_text, clean_lin and clean_author fields
X_whole_vect = scipy.sparse.hstack([X_whole_news_vect, X_whole_link_vect, X_whole_authors_vect])

# Fitting predictive model to the data
y = df["binary_category"].values.astype('U')
err_model = LogisticRegression().fit(X_whole_vect, y)
LogisticRegression()

# Predictions
y_pred = err_model.predict(X_whole_vect)
y_pred_proba = err_model.predict_proba(X_whole_vect)

# Predicted Probabilities for Class 0 and 1
y_pred_proba[:10]

# Create Dataframe of only predictions
df_pred = pd.DataFrame().assign(y = df["binary_category"].values.astype('U'), y_pred = y_pred, y_proba = y_pred_proba[:, 1])

df_pred.head()

from sklearn.metrics import accuracy_score

print("accuracy ", accuracy_score(df_pred.y, df_pred.y_pred))

# Obtain required Indexes (Wrong and Correct predictions)
idxs_correct = df_pred[df_pred.y == df_pred.y_pred].index
idxs_wrong = df_pred[df_pred.y != df_pred.y_pred].index

len(idxs_correct), len(idxs_wrong)
df_pred_correct = df_pred.iloc[idxs_correct]

# High proba & low proba indexes for correct & wrong predictions:
idxs_correct_high = df_pred_correct[df_pred_correct.y_proba > 0.5].index
idxs_correct_low = df_pred_correct[df_pred_correct.y_proba <= 0.5].index

df_pred_wrong = df_pred.iloc[idxs_wrong]
idxs_wrong_high = df_pred_wrong[df_pred_wrong.y_proba > 0.5].index
idxs_wrong_low = df_pred_wrong[df_pred_wrong.y_proba <= 0.5].index

[len(i) for i in [idxs_correct_high, idxs_correct_low, idxs_wrong_high, idxs_wrong_low]]

# Creating new labels based on error info:
df_correct_high = df.iloc[idxs_correct_high, :-1].assign(label = [0 for i in range(len(idxs_correct_high))])
df_correct_low = df.iloc[idxs_correct_low, :-1].assign(label = [1 for i in range(len(idxs_correct_low))])
df_wrong_high = df.iloc[idxs_wrong_high, :-1].assign(label = [2 for i in range(len(idxs_wrong_high))])
df_wrong_low = df.iloc[idxs_wrong_low, :-1].assign(label = [3 for i in range(len(idxs_wrong_low))])
df_correct_high.shape, df_correct_low.shape, df_wrong_high.shape, df_wrong_low.shape

df_error_labels = pd.concat([df_correct_high, df_correct_low, df_wrong_high, df_wrong_low])
#print(df_error_labels.sample(10))

print(df_error_labels.label.value_counts())

# Training new model on error labels:
X_error = df_error_labels.iloc[:, :-1].values
y_error = df_error_labels.iloc[:, -1].values

print(X_error)
print(y_error)
error_model = LogisticRegression().fit(X_whole_vect, y_error)
#print(error_model)

# Four new feats obained, which are to be added to the original data:
error_feats = error_model.predict_proba(X_whole_vect)
#error_feats

print("error features shape ", error_feats.shape, df.shape)

columns1 = ['err_feat-' + str(i + 0) for i in range(1, 5)]
df_error_feats = pd.DataFrame(error_feats, columns = columns1)

#df_error_feats
df = pd.concat([df.iloc[:, :-1], df_error_feats], axis = 1).assign(y = y).round(3)
df.info()

X_whole_vect = scipy.sparse.hstack([X_whole_news_vect, X_whole_link_vect, X_whole_authors_vect, np.array(df["err_feat-1"])[:,None], np.array(df["err_feat-2"])[:,None], np.array(df["err_feat-3"])[:,None], np.array(df["err_feat-4"])[:,None]])

"""**Extracting Train and Test Data**

*    The news data is highly imbalanced and it contains 200,840 documents.  We take 25% of the total data for train and test purpose.  Out of the train and test data, 80% is used for training and the remaining 20% is used for testing.  

*    In machine learning, When we want to train our ML model we split our entire dataset into training set and test set using train_test_split() class present in sklearn.  Then we train our model on training_set and test our model on test_set. This will split the data randomly and the train/test data do not represent the entire data set. This will cause inaccuracy of the models.  To avoid this Stratified sampling is used.  Stratified sample represents the entire dataset in equal proportion.  **StratifiedKFold:** This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class. KFold: Split dataset into k consecutive folds. StratifiedKFold is used when is need to balance of percentage each class in train & test 

*    However StratifiedKFold provides a way to split the entire dataset.  But we need to apply StratifiedKFold for the 25 of the data.  **To achieve this, we used the Group by function with filters.**  The data is grouped based on category labels and 25% of the data is taken from each category.  
"""

#################################################################################
############### Extracting TRAIN AND TEST DATA ##################################
#################################################################################
#Copying index column as groupby creates multi index
df['index1'] = df.index

#dfa = df.loc[df['clean_category'].isin(['POLITICS', 'ENTERTAINMENT', 'WELLNESS'])]
dfa = df

# Grouping the data by clean_category
grouped = dfa.groupby('clean_category', group_keys = True)

# Taking 25% from each category by using sample function.  The final output will have the around 50K rows
# It includes both Train and test data
# Stratified sampling aims at splitting a data set so that each split is similar with respect to category.
trainandtest = grouped.apply(lambda x: x.sample(frac=0.25, replace=False))
print ("train and test data ")
print(trainandtest.clean_category.value_counts())

# Taking 10% data from Train and test data which is around 10K
test = trainandtest.apply(lambda x: x.sample(frac=0.2, replace=False))

#  Taking 90% from Train and test data which is around 40K
df_train = trainandtest.loc[~trainandtest['index1'].isin(test['index1'])]
print ("train data ")
print(df_train.clean_category.value_counts())

# Taking 10% data from Train and test data which is around 5K
df_test = trainandtest.loc[trainandtest['index1'].isin(test['index1'])]
print ("test data ")
print(df_test.clean_category.value_counts())
df_test.head()

"""**Feature Vectors for Models**

The dataset contains headline, description, links, Authors and categories as text features. 

TFIDF is used to get the features for text fields in the dataset.  **Term Frequency-Inverse Document Frequency:** TF-IDF determines how important a word is by weighing its frequency of occurence in the document and computing how often the same word occurs in other documents. If a word occurs many times in a particular document but not in others, then it might be highly relevant to that particular document and is therefore assigned more importance.

We used news_text = headline + short_description, link and author for modeling along with error features which are added based on error analysis (discussed earlier).  **There is more than 10% increase in accuracy by using link, author and error features along with news_text** 
"""

################################################################################ 
#######     FEATURE VECTORS    ################################################
###############################################################################

df_train.info()

# Parameter selection
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 10000

#### Using 3 Vectorizers for building features from news_text, link and authors
vectorizer1 = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

vectorizer2 = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
vectorizer3 = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
vectorizer1.fit_transform(df_train["clean_news_text"].values.astype('U'))
X_train_news_vect = vectorizer1.transform(df_train["clean_news_text"].values.astype('U'))
X_test_news_vect = vectorizer1.transform(df_test["clean_news_text"].values.astype('U'))

vectorizer2.fit_transform(df_train["clean_link"].values.astype('U'))
X_train_link_vect = vectorizer2.transform(df_train["clean_link"].values.astype('U'))
X_test_link_vect = vectorizer2.transform(df_test["clean_link"].values.astype('U'))

vectorizer3.fit_transform(df_train["clean_authors"].values.astype('U'))
X_train_authors_vect = vectorizer3.transform(df_train["clean_authors"].values.astype('U'))
X_test_authors_vect = vectorizer3.transform(df_test["clean_authors"].values.astype('U'))

# Save the vectorizers as a pickle in files
joblib.dump(vectorizer1, (data_path + "/Models" + '/vectorizer_news_text.pkl'))
joblib.dump(vectorizer1, (data_path + "/Models" + '/vectorizer_link.pkl'))
joblib.dump(vectorizer1, (data_path + "/Models" + '/vectorizer_authors.pkl'))


# Build features for train dataset using scipy.sparse.hstack by concatenating TFIDF vectors for news_text
# link, author and error features 
#X_train_vect = scipy.sparse.hstack([X_train_news_vect, X_train_link_vect, X_train_authors_vect])
X_train_vect = scipy.sparse.hstack([X_train_news_vect
                                    , X_train_link_vect, X_train_authors_vect, 
                np.array(df_train["err_feat-1"])[:,None], np.array(df_train["err_feat-2"])[:,None], 
                np.array(df_train["err_feat-3"])[:,None], np.array(df_train["err_feat-4"])[:,None]
                #, np.array(df_train["sentence_count"][:,None])
                ])

#X_test_vect = scipy.sparse.hstack([X_test_news_vect, X_test_link_vect, X_test_authors_vect])
X_test_vect = scipy.sparse.hstack([X_test_news_vect
                                   , X_test_link_vect, X_test_authors_vect, 
                 np.array(df_test["err_feat-1"])[:,None], np.array(df_test["err_feat-2"])[:,None], 
                 np.array(df_test["err_feat-3"])[:,None], np.array(df_test["err_feat-4"])[:,None]
                 #, np.array(df_test["sentence_count"][:,None])
                 ])

y_train = df_train["clean_category_id"]
y_test = df_test["clean_category_id"]

df.info()

"""**Building & Testing ML Models**

After building Feature vectors, we tried with different machine learning classification models in order to find the best modeld that suits the data.  We will try with the following models:

*   Logistic Regression
*   Multinomial Naïve Bayes
*   Linear SVC
*   Random Forest

The methodology used to train each model is as follows:
1.  Step - 1: Decide the hyperparameters that need to be tuned. Execute the models by changing the feature parameters and find the performance
2.  Step - 2: Define the metrics to be used for measuring the performance of the model
  *   Accuracy
      *   Train Accuracy
      *   Test Accuracy
  *   Precision
  *   Recall
  *   F1 Score
  *   Classification Report (precision, recall, f1-score, support)

The dataset contains the following Categories after cleaning
*   POLITICS          
*   WELLNESS          
*   ENTERTAINMENT     
*   PARENTING         
*   STYLE & BEAUTY    
*   TRAVEL            
*   WORLDPOST         
*   FOOD & DRINK      
*   HEALTHY LIVING    
*   QUEER VOICES      
*   BUSINESS          
*   COMEDY            
*   SPORTS             
*   BLACK VOICES       
*   HOME & LIVING      
*   SCIENCE & TECH     
*   ARTS & CULTURE     
*   WOMEN              
*   WEDDINGS           
*   IMPACT             
*   CRIME              
*   DIVORCE            
*   MEDIA              
*   WEIRD NEWS         
*   GREEN              
*   RELIGION           
*   EDUCATION          
*   MONEY              
*   GOOD NEWS          
*   FIFTY              
*   ENVIRONMENT        
*   LATINO VOICES      

As the data is imbalanced, we used Stratified sampling to get the train and test data.

We used 5 algorithms with ensemble models such as **Logistic Regression, Multinominal Naïve Bayes, Linear SVC, Random Forest, and Logistic Regression GridSearchCV** and compared train accuracy, test accuracy scores, precision, recall, and F1 scores.  For this dataset, we found that **Logistic Regression GridSearchCV** showed the best performance compared to the other classifiers.

"""

# Models
#create list of model and accuracy dicts

perform_list = []
def run_model(model_name, est_c, est_pnlty):
    model=''
    filename = ''
    if model_name == 'Logistic Regression':
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        filename = 'lr_model.pkl'
    elif model_name == 'Multinomial Naive Bayes':
        model = MultinomialNB()
        filename = 'mnb_model.pkl'
    elif model_name == 'Linear SVC':
        model = LinearSVC()
        filename = 'lsvc_model.pkl'
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100)
        filename = 'rf_model.pkl'
    elif model_name == 'Logistic Regression GridSearchCV':
        model = LogisticRegression(C=est_c, penalty=est_pnlty, solver='lbfgs', max_iter=2000)      
        filename = 'lr_gsv_model.pkl'
    elif model_name == 'GridSearchCV':
        filename = 'gsv_model.pkl'
        # Create the parameter grid based on the results of random search 
        C = [.0001, .001, .01, .1]
        degree = [3, 4, 5]
        gamma = [1, 10, 100]
        probability = [True]

        param_grid = [
          {'C': C, 'kernel':['linear'], 'probability':probability},
          {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
          {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
        ]

        # Create a base model
        svc = svm.SVC(random_state=8)
        cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
        # Instantiate the grid search model
        mdl = GridSearchCV(estimator=svc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

    oneVsRest = OneVsRestClassifier(model)
    oneVsRest.fit(X_train_vect, y_train)
    y_pred = oneVsRest.predict(X_test_vect)
    y_pred_train = oneVsRest.predict(X_train_vect)

    # Save the model as a pickle in a file
    joblib.dump(oneVsRest, (data_path + "/Models/" + filename))
    
    
    # Performance metrics
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    train_accuracy = round(accuracy_score(y_train, y_pred_train) * 100, 2)
    # Get precision, recall, f1 scores
    precision, recall, f1score, support = score(y_test, y_pred, average='micro')

    print(f'Train Accuracy Score of Basic {model_name}: % {train_accuracy}')
    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
    print(f'Precision : {precision}')
    print(f'Recall    : {recall}')
    print(f'F1-score   : {f1score}')
    print(metrics.classification_report(y_test, y_pred))


    # Add performance parameters to list
    perform_list.append(dict([
        ('Model', model_name),
        ('Train Accuracy', round(train_accuracy, 2)),
        ('Test Accuracy', round(accuracy, 2)),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(f1score, 2))
         ]))

"""**Run Logistic Regression Model**"""

run_model('Logistic Regression', est_c=None, est_pnlty=None)

"""**Run Multinomial Naive Bayes Model**"""

run_model('Multinomial Naive Bayes', est_c=None, est_pnlty=None)

"""**Run Linear SVC**"""

run_model('Linear SVC', est_c=None, est_pnlty=None)

"""**Run Random Forest Model**"""

run_model('Random Forest', est_c=None, est_pnlty=None)

"""**Run GridSearchCV Model**


"""

#  Optimization is done for the model using RandomForest GridCV

param = {'estimator__penalty':['l1', 'l2'], 'estimator__C':[0.001, 0.01, 1, 10]}

opt_mdl = LogisticRegression()
oneVsRest = OneVsRestClassifier(opt_mdl)
oneVsRest.get_params().keys()

# GridSearchCV
kf=KFold(n_splits=10, shuffle=True, random_state=55)
lr_grid = GridSearchCV(oneVsRest, param_grid = param, cv = kf, scoring='f1_micro', n_jobs=-1)
lr_grid.fit(X_train_vect, y_train)
lr_grid.best_params_

run_model('Logistic Regression GridSearchCV',lr_grid.best_params_['estimator__C'],lr_grid.best_params_['estimator__penalty'])

"""**Model Performance after Optimization**

For this dataset, we found that **Logistic Regression GridSearchCV** showed the best performance compared to the other classifiers.


"""

model_performance = pd.DataFrame(data=perform_list)
model_performance = model_performance[['Model', "Train Accuracy", 'Test Accuracy', 'Precision', 'Recall', 'F1']]
model_performance