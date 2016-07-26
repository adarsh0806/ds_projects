# %load poi_id.py
#!/usr/bin/python
import warnings
warnings.filterwarnings('ignore')
import sys
import pickle
sys.path.append("tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
import tester
from tester import dump_classifier_and_data
import pprint
import pandas as pd
from IPython.display import display, HTML
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

##################################################################################################################### 
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 
                     'deferral_payments', 
                     'total_payments', 
                     'loan_advances', 
                     'bonus', 
                     'restricted_stock_deferred', 
                     'deferred_income', 
                     'total_stock_value', 
                     'expenses', 
                     'exercised_stock_options', 
                     'other', 
                     'long_term_incentive', 
                     'restricted_stock', 
                     'director_fees']

email_features = ['to_messages', 
                  'from_poi_to_this_person', 
                  'from_messages', 
                  'from_this_person_to_poi', 
                  'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##################################################################################################################### 
### Task 2: Remove outliers

# Finding employees with 18 or more NaNs in their feature set
for i in data_dict:
    count = 0
    for j in data_dict[i]:
        if data_dict[i][j] == 'NaN':
            count += 1
    if count > 17:
        print 'Name of individual: ',i
        print 'Number of NaN values in the corresponding feature set: ',count

# not the name of a person
data_dict.pop('TOTAL',0) 
# not the name of a person
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0) 
# all 20 features have NaN values
data_dict.pop('LOCKHART EUGENE E',0) 

# Convert data from dictionary to dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=None)

# Remove NaN
salary = df['salary'].replace(['NaN'], 0)
from_poi = df['from_poi_to_this_person'].replace(['NaN'], 0)
bonus = df['bonus'].replace(['NaN'], 0)

# Re shape the data
salary = numpy.reshape( numpy.array(salary), (len(salary), 1))
from_poi = numpy.reshape( numpy.array(from_poi), (len(from_poi), 1))
# bonus = numpy.reshape( numpy.array(bonus), (len(bonus), 1))

# Split the data into training and testing sets to generate a regression line
salary_train, salary_test, from_poi_train, from_poi_test = train_test_split(salary, 
                                                                            from_poi, 
                                                                            test_size=0.1, 
                                                                            random_state=42)

# Using a regression line to view outliers
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg = reg.fit(salary_train, from_poi_train)
reg_pred = reg.predict(salary_test)

# Check accuracy of regression
from sklearn.metrics import r2_score
r = r2_score(from_poi_test, reg_pred)
print 'R score of predicting emails from poi to person in question using the salary: ', r

## No real information gain from plot.
# Plot salary against from_poi
# import matplotlib.pyplot as plt
# %matplotlib inline  
# try:
#     plt.plot(ages, reg.predict(salary), color="blue")
# except NameError:
#     pass
# plt.scatter(salary, from_poi)
# plt.xlabel('Salary')
# plt.ylabel('From POI to this person')
# plt.show()

#####################################################################################################################
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict
my_df = pd.DataFrame.from_dict(my_dataset, orient='index', dtype=None)
my_features_list = ['poi'] + financial_features + email_features

# Clean the data to be used to set up new feature.
salary = my_df['salary'].replace(['NaN'], 0,inplace = True)
total_payments = my_df['total_payments'].replace(['NaN'], 0,inplace = True)
bonus = my_df['bonus'].replace(['NaN'], 0,inplace = True)
total_stock_value = my_df['total_stock_value'].replace(['NaN'], 0,inplace = True)

# Create new column with total monetary assets -> ['net_worth'] using cleaned column data above.
my_df['net_worth'] = my_df['salary'] + my_df['total_payments'] + my_df['bonus'] + my_df['total_stock_value']
new_features_list = my_features_list + ['net_worth']

# Convert dataframe back to dictionary
my_dataset = my_df.to_dict(orient='index')


### Extract features(email and financial) and labels(poi or not) from dataset for local testing

# Takes a list of features ('features_list'), searches the data dictionary for those features, 
# and returns those features in the form of a data list.
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
# Splits the data list, created by the previous statement, into poi(labels) and features
labels, features = targetFeatureSplit(data)

# Use feature selection to select k best features
kbest = SelectKBest(k = 10)
kbest.fit(features, labels)
scores = kbest.scores_

# Combine features with their scores
features_scores = zip(new_features_list[1:], scores)

# Top features
features_scores = dict(features_scores[:21])
sorted_features_scores = sorted(features_scores.items(), key=lambda x: x[1], reverse=True)
best_features = dict(sorted_features_scores[:4]).keys()
best_features = ['poi'] + best_features

# Scale the features                                                             
#MinMax Scaler
scaler = preprocessing.MinMaxScaler()
features = MinMaxScaler().fit_transform(features)

#print new_features_list
print 'POI + best features ', best_features


#####################################################################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


clf = KNeighborsClassifier()
accuracy, precision, recall = [], [], []
for i in range(1000):
    features = MinMaxScaler().fit_transform(features)
    features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                                labels, 
                                                                                test_size=0.3) 
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    # Append scores
    accuracy.append(accuracy_score(labels_test, prediction))
    precision.append(precision_score(labels_test, prediction, average="weighted"))
    recall.append(recall_score(labels_test, prediction, average="weighted"))
print "Classifier details: ", clf
print "Accuracy: ", numpy.mean(accuracy)
print "Precision: ", numpy.mean(precision)
print "Recall: ", numpy.mean(recall)
print '\n'
 
    
#####################################################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Tune K Nearest Neighbors
cv = StratifiedShuffleSplit(labels, 10, random_state = 42)
knn = GridSearchCV(KNeighborsClassifier(), 
                   param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                 'metric': ['manhattan','minkowski', 'euclidean'], 
                                 'weights': ['distance', 'uniform']}, 
                   cv = cv,
                   scoring = 'f1')
knn.fit(features, labels)
print 'K Nearest Neighbors best estimator: ', knn.best_estimator_
print 'K Nearest Neighbors best parameters: ', knn.best_params_
print 'K Nearest Neighbors best score: ', knn.best_score_
# tester.test_classifier(knn.best_estimator_, my_dataset, best_features)

# Pipeline module to run PCA to find features with maximum variance
print "Pipelining..."
pipeline = Pipeline([('normalization', scaler),
                     ('classifier', knn.best_estimator_)
                    ])
tester.test_classifier(pipeline, my_dataset, best_features)

# Best classifier being tested in tester.py
clf =  pipeline
features_list = best_features

#####################################################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)