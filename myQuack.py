
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
import warnings 
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9510761, 'Morgan', 'Frearson'), (1234568, 'FILL', 'ME'), (1234569, 'FILL', 'ME') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    
    # Load the data from the file medical_records.data
    # Assign B or M into the second field 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

    X = np.genfromtxt(dataset_path, delimiter=',', dtype = str, usecols = range(2,31))
    
    y = np.genfromtxt(dataset_path, delimiter=',', dtype = str, usecols = 1)
    
    # Find all the 'M's'
    y_M = np.argwhere(y == 'M')
    
    # Find all the 'B's'
    y_B = np.argwhere(y == 'B')
    
    # Convert 'M' to 1 and 'B' to 0
    y[y_M] = 1
    y[y_B] = 0
    
    # Use Numpy to turn into an array 
    y_field = np.array(y, dtype = int)
    X_field = np.array(X, dtype = float)
    
    return X_field, y_field 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    print('\nClassifier: Decision Tree')
    
    # Create DT classifier using sklearn library
    parameters = {'max_depth':range(1, 100)}
    
    # Use the cross validation provided by GrideSearchCV
    # Add random state to keep results consistent 
    dt_class = GridSearchCV(DecisionTreeClassifier(random_state = 20), parameters)
    
    # Remove the irrelevant warnings 
    warnings.simplefilter("ignore") 
        
    # Train the model with the training data
    dt_class.fit(X_training, y_training)
    
    # Return the automatic cross validated object fitted for the data 
    return dt_class

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    print('\nClassifier: Nearrest Neighbours')
    
    # Create Nearrest Neighbours classifier using sklearn library
    parameters = {'n_neighbors':range(1, 100)}
    
    # Use the cross validation provided by GrideSearchCV
    nnb_class = GridSearchCV(KNeighborsClassifier(), parameters)
    
    # Remove the irrelevant warnings 
    warnings.simplefilter("ignore") 
        
    # Train the model with the training data
    nnb_class.fit(X_training, y_training)
    
    # Return the automatic cross validated object fitted for the data 
    return nnb_class
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    print('\nClassifier: SVM')
    
    # Create SVM classifier using sklearn library
    parameters = {'C':range(1, 100)}
    
    # Remove the irrelevant warnings that affect the F-score label 
    # the F-score is considered to be 0 so the average scores return
    # a warning 
    # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi/47285662
    warnings.filterwarnings('ignore')
    
    # Use the cross validation provided by GrideSearchCV
    # Add a random state for consistent results 
    svm_class = GridSearchCV(SVC(random_state = 20), parameters)
    
    # Train the model with the training data
    svm_class.fit(X_training, y_training)
    
    # Return the automatic cross validated object fitted for the data 
    return svm_class

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network with two dense hidden layers classifier 
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    print('\nClassifier: Neural Network')
    
    # Create Neural Network classifier using sklearn library
    parameters = {'hidden_layer_sizes':range(1, 100)}
    
    # Use the cross validation provided by GrideSearchCV
    # Add a random state to keep results consistent 
    neurN_class = GridSearchCV(MLPClassifier(random_state = 20), parameters)
    
    # Remove the irrelevant warnings 
    warnings.simplefilter("ignore") 
    
    # Train the model with the training data
    neurN_class.fit(X_training, y_training)
    
    # Return the automatic cross validated object fitted for the data 
    return neurN_class

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    print(my_team())
    
    # Medical Data path
    file_path = './medical_records.data'  
    
    # Initalise the data 
    data, label_class = prepare_dataset(file_path)
    
    # Divide the data into training and test sets at ratio 0.8:0.2
    # https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    test_set_size = 0.2
    
    # Divide the training data into training and validation sets at ratio 0.8:0.2
    validation_set_size = 0.2
    
    # Create initial training and testing data set
    X_train, X_test, y_train, y_test = train_test_split(data, label_class, test_size = test_set_size, random_state=5)
    
    # Create the training and validation data set 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= validation_set_size, random_state=5)

    ################## Pick your classifier to run/review ###################
    
    # Comment/Uncomment the classifier you would like to use
    
    classifier = build_DecisionTree_classifier
    #classifier = build_NearrestNeighbours_classifier
    #classifier = build_SupportVectorMachine_classifier
    #classifier = build_NeuralNetwork_classifier
    
    #########################################################################
        
    # Start clock to allow for time testing of classifier 
    start_time = time.clock()
    
    # Build Classifier for the training data 
    clf = classifier(X_train, y_train)
    
    # Output time taken for Cross-validation, optimization and fitting
    print("\nTook: %0.2f seconds For: Cross-validation, optimization and fitting" % (time.clock() - start_time))
    
    # Run the classifier on the training data 
    predictions = clf.predict(X_train)
    
    # Print the best parameters for the data 
    print("\nBest Hyperparameter:", clf.best_params_)
    
    # Print the classification report for training data 
    print("\nClassification Report for 'Training Data':")
    target_names = ['class B (0)', 'class M (1)']
    print(metrics.classification_report(y_train, predictions, target_names=target_names))
    
    # Print the training score 
    print ("\nTraining Score: ", clf.score(X_train, y_train))
    print ("Training MSE: ", metrics.mean_squared_error(y_train, predictions))

    print("########################################################")
    
    # Build Classifier for the validation data 
    predictions = clf.predict(X_val)
    
    # Print the classification report for validation data 
    print("\nClassification Report for 'Validation Data':")
    target_names = ['class B (0)', 'class M (1)']
    print(metrics.classification_report(y_val, predictions, target_names=target_names))
    
    # Print the validation score 
    print ("\nValidation Score: ", clf.score(X_val, y_val))
    print ("Validation MSE: ", metrics.mean_squared_error(y_val, predictions))

    print("########################################################")

    # Build the Classifier for the test data 
    predictions = clf.predict(X_test)
    
    # Print the classification report for test data  
    print("\nClassification Report for 'Test Data':")
    target_names = ['class B (0)', 'class M (1)']
    print(metrics.classification_report(y_test, predictions, target_names=target_names))
    
    # Print the testing score 
    print ("\nTesting Score: ", clf.score(X_test, y_test))
    print ("Testing MSE: ", metrics.mean_squared_error(y_test, predictions))
    
    # Print the confusion matrix to show if the test data was put in the correct class 
    print ("\nTest Data Confusion Matrix: ")
    print (metrics.confusion_matrix(y_test,predictions))

    print("########################################################")

    # Plot the training data 
    print ("########### PLOT TRAINING CROSS-VALIDATION ############")
    param_name = list(clf.param_grid.keys())[0] # Get the parameter used
    param_vals = clf.cv_results_['param_' + param_name] # Grab the parameter data 
    param_corr = clf.cv_results_['mean_test_score'] # Grab the average test score 

#    print (param_vals) # For debugging and to view all data 
#    print (param_corr) # For debugging and to view all data 
    
    # Plot the data (x,y) (Hyperparameter, Mean test score)
    plt.plot(param_vals, param_corr, '-')

    # Configure plot and show
    plt.title("Hyperparameter Optimization by Cross-Validation")
    plt.xlabel(param_name + ' value')
    plt.ylabel('Mean Test Score')
    plt.grid(True)
    plt.show()
