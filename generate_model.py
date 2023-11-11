import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


def model(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=20, random_state=0)
    forest.fit(X_train, y_train)
    print("Random Forest: {0}".format(forest.score(X_train, y_train)))

    return forest


base_data = pd.read_csv("DSP_2.csv")

cols = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
        "Oldpeak", "ST_Slope", "HeartDisease"]
data = base_data[cols].copy()

encoder = LabelEncoder()
data.loc[:, "Sex"] = encoder.fit_transform(data.loc[:, "Sex"])
data.loc[:, "ChestPainType"] = encoder.fit_transform(data.loc[:, "ChestPainType"])
data.loc[:, "RestingECG"] = encoder.fit_transform(data.loc[:, "RestingECG"])
data.loc[:, "ExerciseAngina"] = encoder.fit_transform(data.loc[:, "ExerciseAngina"])
data.loc[:, "ST_Slope"] = encoder.fit_transform(data.loc[:, "ST_Slope"])

y = data.iloc[:, 11]
x = data.iloc[:, 0:11]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

forest = model(X_train, y_train)

my_data = [
    [
        15,  # Age
        1,  # Sex
        1,  # ChestPainType
        120,  # RestingBP
        241,  # Cholesterol
        0,  # FastingBS
        1,  # RestingECG
        146,  # MaxHR
        1,  # ExerciseAngina
        2,  # Oldpeak
        1,  # ST_Slope
    ]
]

print(forest.predict(my_data))

my_data = [
    [
        65,  # Age
        0,  # Sex
        1,  # ChestPainType
        120,  # RestingBP
        150,  # Cholesterol
        0,  # FastingBS
        1,  # RestingECG
        120,  # MaxHR
        1,  # ExerciseAngina
        2,  # Oldpeak
        1,  # ST_Slope
    ]
]

print(forest.predict(my_data))

filename = "model.sv"
pickle.dump(forest, open(filename, 'wb'))
