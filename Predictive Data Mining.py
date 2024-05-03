import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
dataset_path = r"C:\Users\acer1\Downloads\ai&ml\poker-hand-training.csv"
poker_data = pd.read_csv(dataset_path)

# Step 2: Preprocess the data
# No preprocessing required in this case as the dataset is assumed to be clean

# Step 3: Split the data into features and target
X = poker_data.drop(columns=['Poker Hand'])
y = poker_data['Poker Hand']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train and evaluate predictive data mining techniques

# Decision Trees
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_test)
print("Decision Tree Classifier:")
print(classification_report(y_test, dt_y_pred))

# Gaussian Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)
print("Naive Bayes Classifier:")
print(classification_report(y_test, nb_y_pred))

# Neural Network (Multi-layer Perceptron)
mlp_classifier = MLPClassifier(random_state=42)
mlp_classifier.fit(X_train, y_train)
mlp_y_pred = mlp_classifier.predict(X_test)
print("Neural Network Classifier:")
print(classification_report(y_test, mlp_y_pred))

# k-Nearest Neighbors
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_y_pred = knn_classifier.predict(X_test)
print("k-Nearest Neighbors Classifier:")
print(classification_report(y_test, knn_y_pred))

# Random Forest Classifier with parameter tuning
rf_classifier = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_classifier = grid_search.best_estimator_
best_rf_y_pred = best_rf_classifier.predict(X_test)
print("Random Forest Classifier (with parameter tuning):")
print(classification_report(y_test, best_rf_y_pred))

# Step 6: Develop Prototype Mini Classification Tool
def classify_new_data(new_data):
    prediction = best_rf_classifier.predict(new_data)
    return prediction

# Step 7: User Interaction
for i in range(3):
    new_data = input("Enter new data (comma-separated values for features): ")
    new_data = [int(x) for x in new_data.split(',')]
    new_data = [new_data]  # Convert to 2D array
    prediction = classify_new_data(new_data)
    print("Prediction:", prediction)

# Step 8: Generate visualization
conf_matrix = confusion_matrix(y_test, best_rf_y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=poker_data['Poker Hand'].unique(), yticklabels=poker_data['Poker Hand'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
