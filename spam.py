import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r"C:\Users\Admin\Downloads\spam.csv", encoding='ISO-8859-1')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
label_counts = df['label'].value_counts()
label_counts.index = label_counts.index.map({0: 'ham', 1: 'spam'})
print("Dataset label counts:")
print(label_counts)
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
test_label_counts = y_test.value_counts()
test_label_counts.index = test_label_counts.index.map({0: 'ham', 1: 'spam'})
print("\nTest set label counts:")
print(test_label_counts)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print(f'\nAccuracy: {accuracy}')
print('Classification Report:')
print(report)
