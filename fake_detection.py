import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# data set -> news.csv
df = pd.read_csv('news.csv') 
print(df.head()) 

# X-> i/p, Y-> o/p, false-> 1, real-> 0
X = df['text']   
y = df['label']  

# split data-> 1)test set, 2) train set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# tokenization ->(text vectorization)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# model training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# prediction -> test set
y_pred = model.predict(X_test_tfidf)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(accuracy*100, 2)}%')

# Confusion matrix (optional)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

print('Confusion Matrix:')
print(f'True Positives (Fake correctly identified as Fake): {cm[1][1]}')
print(f'True Negatives (Real correctly identified as Real): {cm[0][0]}')
print(f'False Positives (Real wrongly identified as Fake): {cm[0][1]}')
print(f'False Negatives (Fake wrongly identified as Real): {cm[1][0]}')