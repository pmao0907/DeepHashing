import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_data = np.load('CIFAR10-DINOV2-BASE/train.npz')
test_data = np.load('./CIFAR10-DINOV2-BASE/test.npz')
X_train = train_data['embeddings']  # Load embeddings
y_train = train_data['labels']      # Load labels
X_test = test_data['embeddings']    # Load embeddings
y_test = test_data['labels']



print(X_train.shape) # 50,000 samples having embedding dimension of 768

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))