import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import size
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier


def main():
    missing_values = ["n/a", "na", "--"]
    dataset = pd.read_excel (r'clinical_dataset.xlsx',na_values = missing_values)
    status = dataset['Status'] 
    data = dataset.drop(['Status'], axis=1)
    #Summary statistics
    mean = np.mean(data, axis=0)
    standarddeviation = np.std(data, axis=0)
    minimum = np.min(data, axis=0)
    maximum = np.max(data, axis=0)
    #dimensions and size
    dimensions = dataset.shape
    rows = dataset.shape[0]
    columns = dataset.shape[1]
    total = rows * columns
    print("Mean:\n",mean,"\nStandard deviation:\n",standarddeviation,"\nMinimum:\n",minimum,"\nMaximum:\n",maximum)
    print("Rows:",rows," Columns:",columns," Total:",total)
    #Z-score Normalisation
    avg = np.mean(data)  
    std = np.std(data)  
    data_z_score = (data - avg) / std  
    
    BMIhealthy = dataset[dataset['Status'] == 'healthy']['BMI']  
    BMIcancerous = dataset[dataset['Status'] == 'cancerous']['BMI'] 
    #density plots
    sns.kdeplot(BMIcancerous, label="Cancerous status", color="aqua")
    sns.kdeplot(BMIhealthy, label="Healthy status", color="black")
    plt.title('Density plot of BMI')
    plt.xlabel('BMI')
    plt.ylabel('Density')
    plt.show()
    #boxplot
    sns.boxplot(x='Status', y = 'Age', data = dataset)
    plt.show()
    #section 3
    #Re-Normalisation and preparation of data
    df = dataset.sample(frac=1)
    status = df['Status'] 
    df = df.drop(['Status'], axis=1)
    avg = np.mean(df) 
    std = np.std(df)
    inp_z_data = (df - avg) / std  
    status = status.apply(lambda x: 0 if 'healthy' in x else 1)
    #splitting data into test and tran sets
    x_train, x_test, y_train, y_test = train_test_split(inp_z_data, status, test_size=0.1)
    #setting values to be passed to models
    epochs = 1000
    step = 50
    tree_number = [10,50,100, 1000, 5000]
    min_samples = [5, 50]
    #calling functions
    ann_model(x_train, x_test, y_train, y_test, epochs,step)  
    random_forest(x_train, x_test, y_train, y_test, tree_number, min_samples)

    #setting values for cross validation
    hidden_neurons = [50, 500, 1000] 
    trees_number = [20, 500, 10000]  
    #calling cross validation functions
    ann_cv(inp_z_data, status, hidden_neurons, folds=10)  
    rf_cv(inp_z_data, status, trees_number, folds=10)


def ann_model(x_train, x_test, y_train, y_test, epochs,step):
    #create list to store accuracy
    acc_array = []
    for epoch in range(1, epochs, step):
        # Initialising the model
        ann = MLPClassifier(hidden_layer_sizes=[500, 500], activation='logistic',
                            solver='lbfgs', alpha=0.1, max_iter=epoch)     
        #train model           
        ann.fit(x_train, y_train)
        #calculate accuracy from model
        accuracy = ann.score(x_test, y_test)  
        #add accuracy to list
        acc_array.append(accuracy)  
    #output accuracy to screen
    print('ANN final model accuracy: %.2f%%' % (accuracy * 100))
    #plot Accuracy - epochs graph
    plt.plot(range(1, epochs, step), acc_array, color='r', label='Accuracy')  
    plt.xlabel('epochs'), plt.ylabel('accuracy'), plt.title("Epochs - Accuracy Plot")
    plt.legend()
    plt.show()  
    return

def random_forest(x_train, x_test, y_train, y_test, tree_number, min_samples):
    accuracy_5 = []  
    accuracy_50 = [] 
    #create list to store accuracy
    for i in tree_number:
        for ii in min_samples:
            # Initialising the model
            rf = RandomForestClassifier(n_estimators=i, 
            max_depth=None,min_samples_split=ii, bootstrap=True)
            #train model  
            rf.fit(x_train, y_train)  
            #create prediction using model
            y_pred = rf.predict(x_test)  
            #calculate accuracy from model
            accuracy = "%.2f%%" % ((accuracy_score(y_test, y_pred)) * 100)  
            #prepare accuracy so it can be used in graphs
            accuracyf = float(accuracy.rstrip("%"))
            if ii == min_samples[0]:
                #add accuracy to list
                accuracy_5.append(accuracyf)  
                out_accuracy_5 = accuracy
            elif ii == min_samples[1]:
                #add accuracy to list
                accuracy_50.append(accuracyf) 
                out_accuracy_50 = accuracy
    #plot Accuracy - Trees graph
    plt.plot(tree_number, accuracy_5, color='red', label="5 samples")
    plt.xlabel('Trees'), plt.ylabel('accuracy'), plt.title("Trees - Accuracy Plot")
    plt.plot(tree_number, accuracy_50, color='blue', label="50 samples")
    plt.legend()
    plt.show()
    #output accuracy for each sample
    print('Min Samples:50  Forests Model Accuracy:', out_accuracy_50)
    print('Min Samples:05  Forests Model Accuracy:', out_accuracy_5)
    return



def ann_cv(data, status, neurons, folds):
    cross_validation_score = []  
    #create list to store cv score
    for neuron in neurons: 
        k_folds = KFold(shuffle=True, n_splits=folds, random_state=1)
        #create kfolds with raqndom states
        ann = MLPClassifier(hidden_layer_sizes=[neuron, neuron], activation='logistic',
                            solver='lbfgs', alpha=0.1)  
        #Initialise ANN model
        score = cross_val_score(estimator = ann, X = data, y = status, cv = k_folds)  
        #calculate cv score
        cross_validation_score.append(score.mean())
        #add mean score to list and print score
        print("ANN: Neurons:%d Accuracy:%0.2f" % (neuron, score.mean())) 
    #plot CV accuracy for ANN
    plt.plot(neurons, cross_validation_score, color='blue')  
    plt.xlabel('Neurons')
    plt.ylabel('Accuracy')
    plt.title('ANN Accuracy Plot')
    plt.show()  
    return


def rf_cv(data, status, trees, folds):
    #create list to store cv score
    cross_validation_score = []  
    for tree in trees: 
        #create kfolds with raqndom states
        k_folds = KFold(shuffle=True, n_splits=folds, random_state=1)
        #Initialise RF model
        forest_model = RandomForestClassifier(n_estimators=tree, min_samples_split=5,
                                              bootstrap=True) 
        #calculate cv score
        score = cross_val_score(forest_model, data, status, cv=k_folds)  
        #add mean score to list and print score
        cross_validation_score.append(score.mean())
        print("Random Forests: Trees:%d Accuracy:%0.2f" % (tree, score.mean()))  
    #plot CV accuracy for RF
    plt.plot(trees, cross_validation_score, color='r') 
    plt.xlabel('Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forests Accuracy Plot')
    plt.show() 
    return









if __name__ == '__main__':
    """Execute the current module"""
    main()
