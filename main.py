import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("train.csv")

# Feature Engineering
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Lady": "Royalty",
    "the Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Sir": "Royalty",
    "Capt": "Officer",
    "Ms": "Miss"
}
data['Title'] = data['Title'].map(title_mapping)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
data['FarePerPerson'] = data['Fare'] / data['FamilySize']

# Preprocessing
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Title'] = le.fit_transform(data['Title'])

X = data.drop(columns=['Survived'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

best_rfc = RandomForestClassifier(**best_params, random_state=42)
best_rfc.fit(X_train, y_train)
y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
