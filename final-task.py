import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Učitavanje podataka
df = pd.read_csv('products.csv')

# 2. Istraživanje i čišćenje podataka
print(df.isnull().sum())

df = df.dropna(subset=['Product Title', 'Category Label'])

print(df.duplicated().sum())

df = df.drop_duplicates()

# 3. Inženjering karakteristika
df['title_length'] = df['Product Title'].apply(len)
df['num_words'] = df['Product Title'].apply(lambda x: len(x.split()))
df['has_number'] = df['Product Title'].apply(lambda x: any(char.isdigit() for char in x))
df['has_special_char'] = df['Product Title'].apply(lambda x: bool(re.search(r'[^\w\s]', x)))

# 4. Preprocesiranje teksta: Tfidf vektorizer za transformaciju teksta u numeričke vrednosti
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['Product Title'])

# 5. Target (kategorija) - Kategorija proizvoda
y = df['Category Label']

# 6. Splitovanje skupa podataka na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Treniranje modela (Naivni Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Predikcija i evaluacija modela
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 9. Čuvanje modela u .pkl formatu
with open('product_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 10. Skripta za predikciju nove kategorije proizvoda
def predict_category(product_title):
    # Učitaj model i vektorizator
    with open('product_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    product_title_tfidf = vectorizer.transform([product_title])
    
    predicted_category = model.predict(product_title_tfidf)
    return predicted_category[0]

# Testiranje predikcije
test_product = "iphone 7 32gb gold,4,3,Apple iPhone 7 32GB"
predicted_category = predict_category(test_product)
print(f"Predicted Category for '{test_product}': {predicted_category}")

