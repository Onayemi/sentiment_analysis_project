# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:13:27 2024

@author: REMLEX
"""

# To install libraries
# pip install wordcloud
# pip install langdetect
# pip install sumy
# pip install textblob

# Wordcloud - This package is used to generate word cloud images.
# Langdetect - This library is used for language detection.
# Sumy - Automatic summarization of text documents and articles.
# TextBlob - TextBlob is a library for processing textual data

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from langdetect import detect
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from textblob import TextBlob
import seaborn as sns

# Reading the csv file
df = pd.read_csv('chatgpt1.csv')

# Creating a function to detect language
x = df['Text'][0]
lang = detect(x)

def det(x):
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang
    
# Apply this function to all the dataset
df['lang'] = df['Text'].apply(det)
    
# Use the lang to filter language
df = df.loc[df['lang'] == 'en']
df = df.reset_index(drop=True) # To avoid index into column

# Cleaning some text
# df['Text'] = df['Text'].str.replace('https', '')
# df['Text'] = df['Text'].str.replace('http', '')
# df['Text'] = df['Text'].str.replace('t.co', '')

# Developing a sentiment function

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


df['sentiment'] = df['Text'].apply(get_sentiment)


# Generating a word cloud
comment_words = ''
stopwords = set(STOPWORDS)

for val in df.Text:
    val = str(val)
    tokens = val.split()
    comment_words = comment_words + " ".join(tokens)+ " "

wordcloud = WordCloud(width=900, height=500, background_color = 'black', 
                      stopwords = stopwords, min_font_size=10).generate(comment_words)


plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout()
plt.show()

sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.countplot(x = 'sentiment', data=df)
plt.xlabel('Sentiment')
plt.ylabel('Count of Sentiment')
plt.title('Sentiment Distribution')
plt.show()
