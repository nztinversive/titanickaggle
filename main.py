import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv("train.csv")

# Preprocessing
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 1
data['IsAlone'].loc[data['FamilySize'] > 1] = 0

data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId', 'SibSp', 'Parch'], axis=1)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Define features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Train the model with the best parameters
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

# Make predictions and calculate the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
