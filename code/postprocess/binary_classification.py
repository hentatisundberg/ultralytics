
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
import seaborn as sns
from functions import create_connection



def prep_training_data(tracks, valid):
    con = create_connection(tracks)
    tracks = pd.read_sql_query(
    """SELECT * FROM Inference""",
    con)

    # Prepare Valid data and merge with track data
    valid = pd.read_csv(valid, sep = ";")
    valid = pd.merge(valid, tracks, on = "track", how = "left")
    valid = valid[valid['Ledge'].isin(["FAR3", "FAR6", "TRI3", "TRI6"])]
    
    # Remove tracks with one detection and only include annotated tracks 
    valid = valid[valid["nframes"] > 1] 
    valid = valid[valid["Valid"] > -1]

    return(valid)


def train_classifier(dataset):

    # Read annotation data
    #dataset = pd.read_csv(dataset, sep = ";", decimal = ",")
    #dataset = dataset[~dataset["Valid"].isna()]

    # Define response and target
    X_expl = dataset[["track", "start", "end"]]
    X = dataset[["nframes",	"x_first","x_std","x_last","y_first","y_std","y_last", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s", "detect_dens"]]
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

    temp = pd.merge(y_test, X_expl["track"], left_index = True, right_index = True, how = "left").reset_index()
    df_out = pd.merge(temp, df_out, left_index = True, right_index = True)

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()

    print(df_model)
    #print(df_out)
    df_out.to_csv("inference/multimod_valid_nomerge.csv", sep = ";", decimal = ",")
    
    # Save model
    pickle.dump(models['Random Forest'], open("models/unmerged_tracks/RandomForests.sav", 'wb'))
    pickle.dump(models['K-Nearest Neighbor'], open("models/unmerged_tracks/KNearest.sav", 'wb'))
    pickle.dump(models['Naive Bayes'], open("models/unmerged_tracks/NaiveBayes.sav", 'wb'))
    pickle.dump(models['Decision Trees'], open("models/unmerged_tracks/DecisionTrees.sav", 'wb'))
    pickle.dump(models['Support Vector Machines'], open("models/unmerged_tracks/SVM.sav", 'wb'))
    pickle.dump(models['Logistic Regression'], open("models/unmerged_tracks/LogisticRegression.sav", 'wb'))

    
def predict_from_classifier(dataset):
    
    RandFor = pickle.load(open("models/unmerged_tracks/RandomForests.sav", 'rb'))
    KNear = pickle.load(open("models/unmerged_tracks/KNearest.sav", 'rb'))
    NaiveBayes = pickle.load(open("models/unmerged_tracks/NaiveBayes.sav", 'rb'))
    DecisionTree = pickle.load(open("models/unmerged_tracks/DecisionTrees.sav", 'rb'))
    SVM = pickle.load(open("models/unmerged_tracks/SVM.sav", 'rb'))
    LogReg = pickle.load(open("models/unmerged_tracks/LogisticRegression.sav", 'rb'))

    con = create_connection(dataset)
    cond1 = f'nframes > 1'
    
    sql = (f'SELECT * FROM Inference '
            f'WHERE {cond1};')
    
    dataset = pd.read_sql_query(
        sql, 
        con)
    
    preddata = dataset[["nframes","x_first","x_std","x_last","y_first","y_std","y_last", "conf_min","conf_mean","conf_max","mindim_mean","mindim_std","maxdim_mean","maxdim_std","x_dist","y_dist","dur_s", "detect_dens"]]

    # Transform indata
    ss_pred = StandardScaler()
    preddata_transform = ss_pred.fit_transform(preddata)

    # Make predictions 
    pred_randfor=pd.Series(RandFor.predict(preddata_transform), name = "RandFor")
    pred_knear=pd.Series(KNear.predict(preddata_transform), name = "KNear")
    pred_naive=pd.Series(NaiveBayes.predict(preddata_transform), name = "NaBayes")
    pred_dectree=pd.Series(DecisionTree.predict(preddata_transform), name = "DecTree")
    pred_svm=pd.Series(SVM.predict(preddata_transform), name = "SVM")
    pred_logreg=pd.Series(LogReg.predict(preddata_transform), name = "LogReg")

    preds = pd.concat([pred_randfor, pred_knear, pred_naive, pred_dectree, pred_svm, pred_logreg], axis = 1)
    preds["multi"] = preds["RandFor"]+preds["KNear"]+preds["NaBayes"]+preds["DecTree"]+preds["SVM"]+preds["LogReg"]
    preds["fish"] = np.where(preds["multi"] == 6, 1, 0)

    # Combine with original data
    out = pd.merge(preds, dataset, left_index = True, right_index = True)
    
    out.to_csv("inference/Predicted_fishtracks_unmerged.csv", sep = ";", decimal = ",")
    return(out)




def plot_annotations(data, x, y, logx, logy):

    fish = data[data["Valid"] == 1]
    nofish = data[data["Valid"] == 0]

    if logx: 
        x1 = np.log(fish[x])
        x2 = np.log(nofish[x])
    else: 
        x1 = fish[x]
        x2 = nofish[x]

    if logy: 
        y1 = np.log(fish[y])
        y2 = np.log(nofish[y])
    else: 
        y1 = fish[y]
        y2 = nofish[y]

    fig, ax = plt.subplots()
    ax.scatter(x1, y1, c = "r", alpha = .3, label = "Fish", s = 10)
    ax.scatter(x2, y2, c = "b", alpha = .3, label = "No Fish", s = 10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend()
    plt.show()



def plot_results(data, x, y, logx, logy):

    fish = data[data["fish"] == 1]
    nofish = data[data["fish"] != 1]

    if logx: 
        x1 = np.log(fish[x])
        x2 = np.log(nofish[x])
    else: 
        x1 = fish[x]
        x2 = nofish[x]

    if logy: 
        y1 = np.log(fish[y])
        y2 = np.log(nofish[y])
    else: 
        y1 = fish[y]
        y2 = nofish[y]

    fig, ax = plt.subplots()
    ax.scatter(x1, y1, c = "r", alpha = .3, label = "Fish", s = 1)
    ax.scatter(x2, y2, c = "b", alpha = .3, label = "No Fish", s = 1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend()
    plt.show()


def plot_orig_data(date, ledge):

    con = create_connection("inference/Inference_raw_nomerge.db")
    
    cond1 = f'ledge = "{ledge}"'

    sql = (f'SELECT * '
           f'FROM Inference '
           f'WHERE {cond1};')

    dataset = pd.read_sql_query(
        sql,
        con)

    dataset["time2"] = pd.to_datetime(dataset["time2"])
    dataset["date"] = dataset["time2"].dt.date
    dataset = dataset[dataset["date"] == date]
    pred_raw = dataset.merge(inf[["track", "fish"]], on = "track", how = "left")

    fish = pred_raw[pred_raw["fish"] == 1]
    nofish = pred_raw[pred_raw["fish"] != 1]

    trackids = list(fish["track"].unique())
    fish = fish[fish["track"].isin(trackids[0:9])]

    # Plot 1
    fig, ax = plt.subplots()
    ax.scatter(fish["x"], fish["y"], c = "r", alpha = .3, label = "Fish", s = 1)
    #ax.scatter(nofish["x"], nofish["y"], c = "b", alpha = .3, label = "No Fish", s = 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    plt.legend()
    plt.show()

    # Plot 2
    palette = sns.color_palette("bright")
    sns.set(rc = {'axes.facecolor': 'white'})
    ax = sns.lineplot(x= fish["x"], y=fish["y"], hue = fish["track"], palette = palette, sort = False)
    ax.invert_yaxis()
    plt.show()

    return(dataset)



# Prep training data 
valid = prep_training_data("inference/Inference_stats_nomerge.db", "data/fish_track_unmerged_annotations.csv")

# Train model 
train_classifier(valid)

# Predict 
inf = predict_from_classifier("inference/Inference_stats_nomerge.db")

# Plot 
#dataset = plot_orig_data("2022-06-22", "FAR3")

# Plot annotations
plot_annotations(valid, "nframes", "conf_mean", False, False)

# Plot 
#plot_results(inf, "x_dist", "y_dist", True, True)

