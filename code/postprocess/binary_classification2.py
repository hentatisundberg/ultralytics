
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import sqlite3


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def plot_boundary(clf, X, Y):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, edgecolors='k')


def train_classifier(dataset):

    # Read annotation data
    #dataset = pd.read_csv(dataset, sep = ";", decimal = ",")
    #dataset = dataset[~dataset["Valid"].isna()]

    # Define response and target
    X_expl = dataset[["track_id", "start", "end"]]
    X = dataset[["nframes",	"x_first","x_std","x_last","y_first","y_std","y_last", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s"]]
    y = dataset['Valid']

    # Define data sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

    # Save orig data for later validation
    X_expl = X_expl.merge(X, left_index = True, right_index = True)

    # Normalize data
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)

    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    # Train models

    models = {}

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    models['Logistic Regression'] = LogisticRegression()

    # Support Vector Machines
    from sklearn.svm import LinearSVC
    models['Support Vector Machines'] = LinearSVC()

    # Decision Trees
    from sklearn.tree import DecisionTreeClassifier
    models['Decision Trees'] = DecisionTreeClassifier()

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    models['Random Forest'] = RandomForestClassifier()

    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    models['Naive Bayes'] = GaussianNB()

    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    models['K-Nearest Neighbor'] = KNeighborsClassifier()


    from sklearn.metrics import accuracy_score, precision_score, recall_score

    accuracy, precision, recall = {}, {}, {}
    df_out = pd.DataFrame()

    for key in models.keys():
        
        # Fit the classifier
        models[key].fit(X_train, y_train)
        
        # Make predictions
        predictions = models[key].predict(X_test)
        df_out[key] = predictions

        # Calculate metrics
        accuracy[key] = accuracy_score(predictions, y_test)
        precision[key] = precision_score(predictions, y_test)
        recall[key] = recall_score(predictions, y_test)

    temp = pd.merge(y_test, X_expl["track_id"], left_index = True, right_index = True, how = "left").reset_index()
    df_out = pd.merge(temp, df_out, left_index = True, right_index = True)

    # SVM - linear 
    #clf_lin = SVC(kernel='linear').fit(X_train, y_train)
    #accuracy_score(y_test, clf_lin.predict(X_tests))

    # SVM - nonlinear 
    #clf_nonlin = SVC(kernel='poly', degree=2, coef0=1).fit(X_train, y_train)
    #accuracy_score(y_test, clf_nonlin.predict(X_tests))

    # Logistic Regression
    #logmodel = LogisticRegression()

    # fit the model with data
    #logmodel.fit(X_train,y_train)
    
    #predict the model
    #predictions_log=logmodel.predict(X_tests)
    #predictions_svc_lin=clf_lin.predict(X_tests)
    #predictions_svc_nonlin=clf_lin.predict(X_tests)


    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()

    print(df_model)
    print(df_out)
    df_out.to_csv("inference/multimod_valid.csv")
    #print(temp)

    # Caluclate accuracy
    #cm = confusion_matrix(y_test, predictions_svc_nonlin)

    #TN, FP, FN, TP = confusion_matrix(y_test, predictions_svc_nonlin).ravel()

    ##print('True Positive(TP)  = ', TP)
    #print('False Positive(FP) = ', FP)
    #print('True Negative(TN)  = ', TN)
    #print('False Negative(FN) = ', FN)

    # Caluclate accuracy
    #accuracy =  (TP + TN) / (TP + FP + TN + FN)

    #print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))

    # Save model
    #pickle.dump(logmodel, open("models/LogisticBinClass3.sav", 'wb'))
    #pickle.dump(clf_nonlin, open("models/fishtrack_classifier.sav", 'wb'))


def predict_from_classifier(model, dataset):
    
    model = pickle.load(open(model, 'rb'))

    dataset = pd.read_csv(dataset, sep = ";", decimal = ",")
    preddata = dataset[["nframes","x_first","x_std","x_last","y_first","y_std","y_last", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s"]]

    # Transform indata
    ss_pred = StandardScaler()
    preddata_transform = ss_pred.fit_transform(preddata)

    # Make predictions 
    predictions_full=pd.Series(model.predict(preddata_transform), name = "Pred")

    # Combine with original data
    out = pd.merge(predictions_full, dataset, left_index = True, right_index = True)

    return(out)


def plot_results(data, x, y):

    out = data
    fish = out[out["Pred"] == 1]
    nofish = out[out["Pred"] != 1]

    fig, ax = plt.subplots()
    ax.scatter(fish[x], fish[y], c = "r", alpha = .3, label = "Fish")
    ax.scatter(nofish[x], nofish[y], c = "b", alpha = .3, label = "No Fish")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend()
    plt.show()



# Train
# Read track data
con = create_connection("inference/Inference.db")
tracks = pd.read_sql_query(
    """SELECT * FROM Inference""",
    con)

# Read Valid data and merge with track data
valid = pd.read_csv("inference/tracks_valid.csv", sep = ";")
valid = pd.merge(valid, tracks, on = "track_id", how = "left")
valid = valid[valid['Ledge'].isin(["FAR3", "FAR6", "TRI3", "TRI6"])]

# Train model 
train_classifier(valid)

# Predict 
out = predict_from_classifier("models/fishtrack_classifier.sav", "inference/tracks_annotate.csv")

# Plot 
# plot_results(out, "x_std", "y_std")
