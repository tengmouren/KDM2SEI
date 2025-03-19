import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load training and test data
mat_1 = scipy.io.loadmat(r"feature_dataset/feature_train_10Class.mat")
X_train = mat_1["f_data"][:, -28:, 0]  # Assuming last 28 features are BPSK
labels = np.squeeze(mat_1["label"])

mat_data_test = scipy.io.loadmat(r"feature_dataset/feature_test_10Class.mat")
X_test = mat_data_test["f_data"][:, -28:, 0]
Y_test = np.squeeze(mat_data_test["label"])

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use the same scaler as the training set

# Define classifiers
clf1 = RandomForestClassifier(n_estimators=50, random_state=1)
clf2 =AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    algorithm="SAMME.R",
    random_state=1
)
clf3 = KNeighborsClassifier(n_neighbors=3)
clf4 = SVC(probability=True, kernel='linear', random_state=1)

# Combine them using a voting classifier
eclf = VotingClassifier(estimators=[
    ('rf', clf1), ('adb', clf2), ('knn', clf3), ('svc', clf4)],
    voting='soft')

# Train the ensemble classifier
eclf.fit(X_train, labels)

# Predict on test set
y_pred = eclf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Test Set Accuracy: ", accuracy)
