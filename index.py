import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset into a pandas dataframe
data = pd.read_csv('customer_purchases.csv')
print(data)

# Use clustering technique to group customers based on their purchase behaviour.
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
clusters = kmeans.predict(data)

# Use classification techniques to predict which products a customer is likely to purchase based on their cluster membership. 
# We'll split the data into training and testing sets, and use a decision tree classifier to make predictions:
X_train, X_test, y_train, y_test = train_test_split(data, clusters, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Use the predictions to recommend products to customers
recommended_products = data[clusters == 1].mean().sort_values(ascending=False)[:5].index.tolist()
print("Recommended products for cluster 1 customers: ", recommended_products)
