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

## Customization
- Change `max_depth` to **increase or decrease tree complexity**.
- Use `criterion="entropy"` instead of `gini` for **information gain-based splitting**.
- Adjust `test_size` in `train_test_split` to modify train-test ratio.

## Conclusion
This project provides a simple yet effective implementation of a **Decision Tree Classifier**. The Iris dataset serves as a great starting point to understand decision trees, their visual representation, and how they classify data.





