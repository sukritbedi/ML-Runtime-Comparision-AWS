import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pylab as plt
from matplotlib import colors
import timeit

import seaborn as sns
sns.set(color_codes=True)

from sklearn.datasets.samples_generator import make_blobs

def train_test_split(data_set, features, labels, train_fraction = 0.8):
    """
      
    """
    numerical_features = features['numerical']
    categorical_features = features['categorical']
    feature_matrix = None
    
    if len(categorical_features) == 0 :        
        feature_matrix = data_set[numerical_features].values
    elif len(numerical_features) == 0 :
        new_data_frame = pd.concat(pd.get_dummies(data_set[categorical_features]), axis=1)
        feature_matrix = new_data_frame.values
    else:
        new_data_frame = pd.concat([data_set[numerical_features],
                                   pd.get_dummies(data_set[categorical_features])], axis=1)
        feature_matrix = new_data_frame.values
        
    lables_array = data_set[labels].values    
    
    shuffled_indices = np.linspace(0, lables_array.shape[0]-1, lables_array.shape[0], dtype=int)    
    np.random.shuffle(shuffled_indices)
    
    feature_matrix = feature_matrix[shuffled_indices, :]
    lables_array = lables_array[shuffled_indices]
    
    train_upper_limit = (int)(lables_array.shape[0]*train_fraction)
    
    train_features = feature_matrix[0:train_upper_limit, :]
    train_labels = lables_array[0:train_upper_limit]
    
    test_features = feature_matrix[train_upper_limit:, :]
    test_labels = lables_array[train_upper_limit:]
    
    return (train_features, train_labels, test_features, test_labels)

#Start timer
start =  timeit.default_timer()

## running some test cases
test_data = {'feature_1': pd.Series([1,2,3,4,5,6,7,8,9,10]),
             'feature_2': pd.Series([11,12,13,14,15,16,17,18,19,20]),
              'label' :  pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])}
test_frame = pd.DataFrame(test_data)
np.random.seed(1024)

test_features = {'numerical': ['feature_1', 'feature_2'], 'categorical' :[]}
t1, t2, t3, t4 = train_test_split(test_frame,test_features , 'label', 0.8)
assert t1.shape[0] == 8 and t2.shape[0] == 8 and \
       t3.shape[0] == 2 and t4.shape[0] == 2,  "train test splitting is not correct"
npt.assert_array_almost_equal(t1, np.array([[3,13], [4, 14], [8, 18], [1, 11], [7, 17], [9, 19], \
                                     [6, 16], [5, 15]]))

test_features = {'numerical': ['feature_1', 'feature_2'], 'categorical' :[]}
t1, t2, t3, t4 = train_test_split(test_frame, test_features, 'label', 0.5)
assert t1.shape[0] == 5 and t2.shape[0] == 5 and \
       t3.shape[0] == 5 and t4.shape[0] == 5,  "train test splitting is not correct"
    
test_data = {'feature_1': pd.Series([1,2,3,4,5,6,7,8,9,10]),
             'feature_2': pd.Series(['A', 'C', 'B', 'A', 'A', 'C', 'B', 'B', 'C', 'C']),
              'label' :  pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])}
test_frame = pd.DataFrame(test_data)
test_features = {'numerical': ['feature_1'], 'categorical' :['feature_2']}
t1, t2, t3, t4 = train_test_split(test_frame, test_features, 'label', 0.5)
assert t1.shape[0] == 5 and t2.shape[0] == 5 and \
       t3.shape[0] == 5 and t4.shape[0] == 5,  "train test splitting is not correct"


def get_k_fold_indices(no_of_data_rows, no_of_folds = 5):
    if no_of_data_rows <= 0:
        raise Exception('no_of_data_rows can\'t be zero or negative')
    
    if no_of_folds <= 0:
        raise Exception('no_of_folds can\'t be zero or negative')
        
    if no_of_folds > no_of_data_rows:
        raise Exception('no_of_folds can\'t be greater than no_of_data_rows')
    
    for i in range(no_of_folds):
        start_index = (no_of_data_rows*i)/no_of_folds
        stop_index = (no_of_data_rows*(i+1))/no_of_folds -1 
        yield (start_index, stop_index)

##for start, end in get_k_fold_indices(101, 5):
    ##print start, end


def predict_probability(features, coefficients):
    product = np.dot(features, coefficients)
    proba = 1.0 / ( 1.0 + np.exp(-product))
    return proba

## some test cases
predict_probability(np.array([1,2,3]), np.array([1,2,3]))

def feature_derivatives(errors, features):
    derivatives = np.dot(np.transpose(features), errors)
    return derivatives

errors = np.array([1,2,3])
feature_matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
feature_derivatives(errors, feature_matrix)

def calculate_negative_log_likelihood(labels, coefficients, feature_matrix):
    #indicator = (labels == 1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1 + np.exp(-scores))
    
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    likelihood = np.sum((1 - labels)*scores + logexp)
    return likelihood

labels = np.array([1,1,0,1])
coefficients = np.array([1,0.5,1])
feature_matrix = np.array([[1,2,3], [3,2,1], [3,3,3], [6,1,2]])
calculate_negative_log_likelihood(labels, coefficients, feature_matrix)
def logistic_regression(feature_matrix, labels, step_size, max_iter, debug = True):
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]), feature_matrix))
    coefficients = np.zeros(feature_matrix.shape[1])
    for itr in xrange(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        errors = predictions - labels
        
        coefficients -= step_size*feature_derivatives(errors, feature_matrix)
        # Checking whether log likelihood is increasing
        if debug and (itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0):
            lp = calculate_negative_log_likelihood(labels, coefficients, feature_matrix)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)        
    return coefficients

## Let's run a small test case
labels = np.array([1,1,0,1])
coefficients = np.array([1,0.5,1])
feature_matrix = np.array([[1.0,2,3], [1,2,1], [1.0,3,3], [1,1,2.0]])

coef = logistic_regression(feature_matrix, labels, 0.1, 1000)

def logistic_prediction(feature_matrix, coefficients, cutoff = 0.5):
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]), feature_matrix))
    probabilities = predict_probability(feature_matrix, coefficients)
    return (probabilities, probabilities > cutoff)

y_pred = logistic_prediction(feature_matrix, coef)[1]
print 'actual: %s' %(labels.astype(bool))
print 'predicted: %s' %(y_pred)



n_samples = 1500

X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, \
    cluster_std=0.5, random_state=0)

#_, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

#ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', s=40, cmap=plt.cm.Spectral)
  
coef = logistic_regression(X, y, 0.02, 750, False)

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 100))
Z = logistic_prediction(np.c_[xx.ravel(), yy.ravel()], coef)[1]
Z = Z.reshape(xx.shape)

#ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
#ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', cmap=plt.cm.Spectral, s=40)

#plt.xlim(X[:, 0].min(), X[:, 0].max())
#plt.ylim(X[:, 1].min(), X[:, 1].max())
#plt.show()

col_names = ['age', 'workclass', 'fnlwgt', 'education', 
             'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '<=50K']

data = pd.read_csv('./data/adult.data', index_col=False, names = col_names)
data.head(5)

data['target'] = data['<=50K'].apply(lambda x : 0 if x.strip() == '<=50K' else 1)
data = data.drop('<=50K', axis=1)

response_variable = data['target']
print 'No: of zeor responses: %d' %sum(response_variable == 0)
print 'No of one responses: %d' %sum(response_variable == 1)

## If we create a simple model which always predicts zero, you will get
print 'Performance of always zero model: %f' %(float(sum(response_variable == 0))/ 
                                              (sum(response_variable == 0) + sum(response_variable == 1)))


numerical = ['age', 'education-num', 'hours-per-week']
categorical = ['education', 'workclass', 'race', 'sex', 'occupation', 'relationship', 'native-country']
simple_features = {'numerical': numerical, 'categorical' :categorical}
train_featues, train_lables, test_features, test_labels = train_test_split(data, 
                                                                          simple_features, 'target', 0.7)
coefficients = logistic_regression(train_featues, train_lables, 1.0e-8, 75000)
test_prediction =  logistic_prediction(test_features, coefficients)[1]

def prediction_accuracy(actual, predicted):
    return sum(actual == predicted)/ float(len(actual))

print 'prediction accuracy (test set): %f' %(prediction_accuracy(test_labels, test_prediction))

traing_size = train_featues.shape[0]
learning_rates = [0.25e-6, 1.0e-7, 0.75e-7, 0.5e-7, 0.25e-7, 1.0e-8]
average_accuracies = []

best_cv_accuracy = 0.0
best_learning_rate = 0.0

for learning_rate in learning_rates:
    fold_accuracies = []
    for start_idx, end_idx in get_k_fold_indices(traing_size, 3):
        cv_test_features = train_featues[start_idx:end_idx+1]
        cv_train_features = np.row_stack((train_featues[0:start_idx], train_featues[end_idx+1:traing_size]))
        assert cv_test_features.shape[0] + cv_train_features.shape[0] == traing_size
        
        cv_test_labels = train_lables[start_idx:end_idx+1]
        cv_train_lables = np.hstack((train_lables[0:start_idx], train_lables[end_idx+1:traing_size]))
        assert cv_test_labels.shape[0] + cv_train_lables.shape[0] == traing_size
        
        coefficients = logistic_regression(cv_train_features, cv_train_lables, learning_rate, 10000, debug=False)
        cv_prediction =  logistic_prediction(cv_test_features, coefficients)[1]
        accuracy = prediction_accuracy(cv_test_labels, cv_prediction)
        fold_accuracies.append(accuracy)
        print 'prediction accuracy (cv set): %f' %(accuracy)
    
    average_accuracy = sum(fold_accuracies)/float(len(fold_accuracies))
    average_accuracies.append(average_accuracy)
    if best_cv_accuracy <  average_accuracy:
        best_cv_accuracy = average_accuracy
        best_learning_rate = learning_rate
    print 'average accuracy: %f for learning rate: %e' %(average_accuracy, learning_rate)
    
print '\nbest accuracy: %f for learning rate: %e' %(best_cv_accuracy, best_learning_rate)  

#plt.xlim(0, 0.26e-6)
#plt.ylim(0.7, 0.9)
#plt.title('Learning reate vs. Prediction accuracy')
#plt.scatter(learning_rates, average_accuracies, c='green')
#plt.plot(learning_rates, average_accuracies, c='green')
#plt.show()

print 'training the model with full dataset using the best hyper-parameters'
coefficients = logistic_regression(train_featues, train_lables, best_learning_rate, 75000)
test_prediction =  logistic_prediction(test_features, coefficients)[1]

#Stop timer
stop = timeit.default_timer()

print 'prediction accuracy (test set): %f' %(prediction_accuracy(test_labels, test_prediction))
print ' Time Elapsed: ', stop-start
