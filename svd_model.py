# Load libraries
#Data Preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pickle


df = pd.read_csv("./spotify-2023.1.csv")


predict = 'mode'

spotify_data = df.copy()
x = np.array(df.drop(predict, axis=1)) 
y = np.array(df[predict]) 
le = preprocessing.LabelEncoder()
bpm = le.fit_transform(list(spotify_data["bpm"]))
key_mapping = {
    'C': 0,
    'C#': 1,
    'D': 2,
    'D#': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'G#': 8,
    'A': 9,
    'A#': 10,
    'B': 11
}
spotify_data["key_encoded"] = spotify_data["key"].map(key_mapping)
mode = le.fit_transform(list(spotify_data["mode"]))
danceability = le.fit_transform(list(spotify_data["danceability_%"]))
valence = le.fit_transform(list(spotify_data["valence_%"]))
energy = le.fit_transform(list(spotify_data["energy_%"]))
acousticness = le.fit_transform(list(spotify_data["acousticness_%"]))
instrumentalness = le.fit_transform(list(spotify_data["instrumentalness_%"]))
speechiness = le.fit_transform(list(spotify_data["speechiness_%"]))


X = list(zip(danceability, valence, energy, acousticness, instrumentalness, speechiness, bpm, spotify_data["key_encoded"]))
y = list(mode)

num_folds = 10
seed = 7
scoring = 'accuracy'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

np.shape(X_train), np.shape(X_test)

X_array = np.array(X)
X_array

models = [('HDBC', HistGradientBoostingClassifier())]
results = []
names = []
print("Performance on Training set")
for name, model in models:
    kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg += '\n'
    print(msg)

fig = plt.figure()
fig.suptitle('HistGradientClassifier Box Plot')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

best_model = HistGradientBoostingClassifier()

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

def predict (features):
    return best_model.predict(features)

features_to_predict = np.array([55, 85, 65, 31, 0, 2, 56, 2]).reshape(1, -1)
prediction = best_model.predict(features_to_predict)

#Classification Report
print(classification_report(y_test, y_pred))


#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Prediction report
for x in range(len(y_pred)):
    print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", X_test[x],)

# Save the trained model to a file
model_file = 'svdmodel.h5'
with open(model_file, 'wb') as file:
    pickle.dump(best_model, file)

# Load the trained model from the file
if os.path.exists(model_file):
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
else:
    print(f"Error: {model_file} does not exist")

# Score the loaded model with test data
score = loaded_model.score(X_test, y_test)
print(f"R-squared value: {score}")