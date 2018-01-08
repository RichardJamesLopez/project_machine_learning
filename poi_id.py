#!/usr/bin/python

import sys
import pickle
import pandas as pd
import pprint
import matplotlib.pyplot

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester_v2 import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'pct_poi_inbound',
                 'pct_poi_outbound']

financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other',
                      'long_term_incentive', 'restricted_stock',
                      'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person',
                  'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']
email_features.remove('email_address')



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

#pprint.pprint(data_dict['TOTAL'])
data_dict.pop('TOTAL', 0)

# Data Imputation
# df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=float)

# for feature in email_features:
#     df[feature].fillna(df[feature].mean(), inplace=TRUE)
# for feature in financial_features:
#     df[feature].fillna(0,inplace=TRUE)


### Task 3: Create new feature(s)

my_dataset = data_dict

def compute_fraction( numerator, denominator):
    if numerator == 'NaN' or denominator == 'NaN':
        fraction = 0
    else:
        fraction = float(numerator)/float(denominator)
    return round(fraction, 2)


def add_fraction_to_dict(dict, numerator, denominator, new_variable_name):
    num = dict[numerator]
    den = dict[denominator]
    fraction = compute_fraction(num, den)
    dict[new_variable_name] = fraction
    return dict

my_dateset = data_dict

for p in my_dataset:
    # calculate inbound POI email fractions
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],                   'from_poi_to_this_person',
    'to_messages',
    'fraction_from_poi')

    # calculate outbound POI email fractions
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
    'from_this_person_to_poi',
    'from_messages',
    'fraction_to_poi')

    # calculate Salary as fraction of Total Payments
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
    'salary',
    'total_payments',
    'fraction_salary_total_payments')

email_features = email_features + ['fraction_from_poi', 'fraction_to_poi']
financial_features = financial_features + ['fraction_salary_total_payments']

features_list = ['poi'] + financial_features + email_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
## libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC #SVC or SVM?
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif



### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
algorithms = ['Naive_Bayes',
#                'SVC',
#                'Standard_Decision_Tree',
#                'K_Nearest_Neighbors',
#                'Adaboost',
#                'Random_Forest',
#                'LinearDiscriminantAnalysis'
                ]


def create_classifier_step(algorithm):
    cl_params = {}
    if algorithm == 'Naive_Bayes':
        cl = GaussianNB()
    elif algorithm == 'SVC':
        cl = SVC()
        cl_params = { algorithm + '__kernel' : ['rbf', 'poly'],
                      algorithm + '__C' : [1000, 10000, 100000]
                    }
    elif algorithm == 'Standard_Decision_Tree':
        cl = tree.DecisionTreeClassifier()
        cl_params = { algorithm + '__min_samples_split' : [2] }
    elif algorithm == 'K_Nearest_Neighbors':
        cl = KNeighborsClassifier()
        cl_params = { algorithm + '__n_neighbors' : [6, 8, 10],
                      algorithm + '__weights' : ['uniform']
                    }
    elif algorithm == 'Adaboost':
        cl = AdaBoostClassifier()
        cl_params = { algorithm + '__n_estimators' : [5, 8, 10, 20, 30, 50, 100],
                      algorithm + '__learning_rate' : [0.025, 0.05, 0.1, 0.5, 1, 2, 4, 6]
                    }
    elif algorithm == 'Random_Forest':
        cl = RandomForestClassifier()
        cl_params = { algorithm + '__max_features' : ['sqrt', 'log2'],
                      algorithm + '__n_estimators' : [2, 5, 7, 10, 15]}
    elif algorithm == 'LinearDiscriminantAnalysis':
        cl = LinearDiscriminantAnalysis()
        cl_params = { algorithm + '__solver' : ['lsqr']}
    return (algorithm, cl), cl_params

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# libraries
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# compare the algorithms

algorithm_comparison = [['ALGORITHM',
                            'ACCU',
                            'PREC',
                            'RECA',
                            'F1',
                            'F2']]
for a in algorithms:
    classifier_step, clf_step_params = create_classifier_step(a)
    print 'Now Running', classifier_step[0].upper()

    min_max_scaler = MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(features_train)


    pipe = Pipeline(steps=[
                            ('MMS', MinMaxScaler()),
                            ('SKB', SelectKBest()),
                            classifier_step])

    params = {
            'SKB__k' : [6,7,8,9,10,11,12],
            'SKB__score_func' : [f_classif]
            }

params.update(clf_step_params)

sss= StratifiedShuffleSplit(labels_train, n_iter = 20, test_size = 0.5, random_state=0)

gscv = GridSearchCV(pipe,
                    params,
                    verbose = 0,
                    scoring = 'f1_weighted',
                    cv=sss)

gscv.fit(features_train, labels_train)

pred = gscv.predict(features_test)

clf = gscv.best_estimator_


# print statements
print a, "performancereport: "
print (classification_report(labels_test, pred))

print 'Now running test_classifier...'


acc, prec, rec, f1, f2 = test_classifier(clf, my_dataset, features_list)

algorithm_comparison.append([a,
                            "{0:.2f}".format(acc),
                            "{0:.2f}".format(prec),
                            "{0:.2f}".format(rec),
                            "{0:.2f}".format(f1),
                            "{0:.2f}".format(f2)
                            ])

for algo in algorithm_comparison:
    print algo[0].ljust(22), algo[1], algo[2], algo[3], algo[4]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
