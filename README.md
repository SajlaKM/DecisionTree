# DecisionTree
ML Projects using Decision Tree Classifier
# Decision Tree Classifier using the Iris Dataset

## Overview
This project demonstrates the implementation of a **Decision Tree Classifier** using the **Iris dataset** in Python. The model is built using the `scikit-learn` library and visualized using `plot_tree` to understand the decision-making process of the classifier.

## Dataset Information
The **Iris dataset** is a well-known dataset in machine learning that contains 150 samples of iris flowers, with:
- **4 features:** Sepal length, Sepal width, Petal length, Petal width
- **3 classes:** Setosa, Versicolor, Virginica

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Implementation
### **1. Load the Dataset**
```python
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels
```

### **2. Train-Test Split**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **3. Train the Decision Tree Model**
```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X_train, y_train)
```

### **4. Visualize the Decision Tree**
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

### **5. Model Evaluation**
```python
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Expected Output
- A **decision tree visualization** showing the classification rules.
- An **accuracy score** that evaluates the model performance.


# Decision Tree Regressor for House Price Prediction

## Overview
This project demonstrates the implementation of a **Decision Tree Regressor** to predict house prices based on synthetic data. The model is built using the `scikit-learn` library and visualized using `plot_tree` to understand the decision-making process of the regressor.

## Dataset Information
For simplicity, we generate synthetic house price data using NumPy. The dataset consists of:
- **1 feature:** Randomly generated house-related numerical data
- **Target variable:** House prices with added noise

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Implementation
### **1. Generate the Dataset**
```python
import numpy as np

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)  # Random features
y = np.sin(X).ravel() + np.random.randn(80) * 0.1  # Target with noise
```

### **2. Train-Test Split**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **3. Train the Decision Tree Regressor**
```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(max_depth=4)
regressor.fit(X_train, y_train)
```

### **4. Visualize the Decision Tree**
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(regressor, filled=True)
plt.show()
```

### **5. Model Evaluation**
```python
from sklearn.metrics import mean_squared_error

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
```

## Expected Output
- A **decision tree visualization** showing the regression splits.
- A **mean squared error score** that evaluates model performance.

- ## Customization
- Change `max_depth` to **increase or decrease tree complexity**.
- Use `criterion="entropy"` instead of `gini` for **information gain-based splitting**.
- Adjust `test_size` in `train_test_split` to modify train-test ratio.
- Change `max_depth` to **increase or decrease tree complexity**.
- Adjust `test_size` in `train_test_split` to modify train-test ratio.
- Experiment with different **feature generation methods** to simulate real-world data.

 

## Conclusion
This project provides a simple yet effective implementation of a **Decision Tree Classifier**. The Iris dataset serves as a great starting point to understand decision trees, their visual representation, and how they classify data and provides a simple yet effective implementation of a **Decision Tree Regressor**. It demonstrates how decision trees can be used for regression tasks, including house price prediction.

---
**Author:** Sajla KM  
**License:** MIT





