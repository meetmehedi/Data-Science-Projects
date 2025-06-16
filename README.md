# Tweet Sentiment Analysis
 - Jupyter Notebook Style

# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# 📥 Step 2: Load the Dataset
nltk.download('stopwords')
df = pd.read_csv("Tweets.csv")
df = df[['text', 'airline_sentiment']]

# 🧹 Step 3: Clean and Preprocess Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# 🔠 Step 4: Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['airline_sentiment']

# 🔀 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Step 6: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📊 Step 7: Evaluation
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 💾 Step 8: Save Model and Vectorizer
-pickle.dump(model, open("sentiment_model.pkl", "wb"))
-pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# 📈 Model Performance
# ✅ Accuracy: 75%

# 💡 Suggestions for Improvement:
 1. Try Logistic Regression, Random Forest, or XGBoost for better accuracy.
 2. Use GridSearchCV to tune TF-IDF vectorizer (ngram_range, max_df, min_df).
 3. Apply advanced preprocessing (lemmatization with spaCy, emoji/text normalization).
 4. Integrate a simple Streamlit UI for real-time tweet input and prediction.
 5. Experiment with BERT or Transformer-based models for state-of-the-art results.
 6. Deploy using Flask API and host on Render, Hugging Face, or Vercel.

