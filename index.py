import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

dataset = pd.read_csv("diabetes_prediction_dataset.csv")

dataset.head()

dataset.info()

X = dataset.drop(columns=["diabetes"])
y = dataset.diabetes

print(X)
print("-----------------")
print(y)

dataset.diabetes.value_counts()

categorical_ix = X.select_dtypes(include=['object']).columns
t = [('cat', OneHotEncoder(), 
      categorical_ix)]
col_transform = ColumnTransformer(transformers=t)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

dt = tree.DecisionTreeClassifier(random_state=1, max_depth=9)

pipeline = Pipeline(steps=[('prep',col_transform),
                           ('model',dt)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred, average=None)

accuracy = accuracy_score(y_test, y_pred)
print("%.2f" % accuracy)
print(f1)

knn = KNeighborsClassifier(n_neighbors = 5)

pipeline_knn = Pipeline(steps=[('prep',col_transform),
                           ('model',knn)])

pipeline_knn.fit(X_train, y_train)

f1 = f1_score(y_test, y_pred, average="macro")

y_pred_knn = pipeline_knn.predict(X_test)
acc_knn = accuracy_score(y_test,y_pred_knn)
print("%.2f" % acc_knn)
print(f1)

