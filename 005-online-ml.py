from sklearn.linear_model import SGDClassifier
import numpy as np

# Step 1: Create model (supports online learning)
model = SGDClassifier()

# Step 2: Initial training (small dataset)
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 1])

model.partial_fit(X_train, y_train, classes=[0, 1])

# Step 3: New data arrives (streaming)
X_new = np.array([[4, 5]])
y_new = np.array([1])

# Step 4: Update model with new data
model.partial_fit(X_new, y_new)

# Step 5: Make prediction
prediction = model.predict([[5, 6]])
print("Prediction:", prediction)