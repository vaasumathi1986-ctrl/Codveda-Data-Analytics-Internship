import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


df = pd.read_csv("3) Sentiment dataset.csv")


print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows of Text & Sentiment Columns:")
print(df[['Text', 'Sentiment']].head(5))
print("\n--------------------------------------------\n")

stop_words = set(stopwords.words('english'))
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|[^a-zA-Z\s]', '', text.lower())
    return " ".join([w for w in text.split() if w not in stop_words])

df['Cleaned_Text'] = df['Text'].apply(clean_text)

print(df['Sentiment'].value_counts())

plt.figure(figsize=(9, 5))

top_10_sentiments = df['Sentiment'].value_counts().head(10)
sns.barplot(x=top_10_sentiments.values, y=top_10_sentiments.index, palette='viridis')

plt.title("Top 10 Sentiment Distribution")
plt.xlabel("Number of Reviews Count")
plt.ylabel("Sentiment Categories")
plt.tight_layout() 
plt.show()

all_words = " ".join(df['Cleaned_Text'])
wordcloud = WordCloud(
    width=1000, 
    height=500, 
    background_color='white', 
    max_words=100,
    scale=2,           
    max_font_size=150,
    colormap='plasma'  
).generate(all_words)
plt.figure(figsize=(9, 4), dpi=100) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.title("Most Frequent Review Words Cloud (HD)", fontsize=16)
plt.tight_layout()
plt.show()