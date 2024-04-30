import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = np.array(data_dict['data2'])
labels = np.array(data_dict['labels'])

# Preprocess the data to ensure all elements have the same length
max_length = max(len(item) for item in data)
data = np.array([np.pad(item, (0, max_length - len(item)), 'constant', constant_values=0) for item in data])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Apply feature selection to both training and testing data
selector = SelectKBest(score_func=f_classif, k=84)  # Select top 84 features based on ANOVA F-value
train1 = selector.fit_transform(x_train, y_train)
test1 = selector.transform(x_test)  # Use transform instead of fit_transform for the testing data

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(train1, y_train)

# Make predictions on the testing data
y_predict = model.predict(test1)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
