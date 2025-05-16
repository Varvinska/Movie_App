#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


st.title("🎬 Movie Insight Dashboard")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Varvinska/Movie_App/refs/heads/main/movie_dataset.csv'
    return pd.read_csv(url)

df = load_data()

# Preview
if st.checkbox("Show raw data"):
    st.dataframe(df.head(20))

# Genre Analysis
st.subheader("🍿 Most Rated Genres")

genre_counts = df['genres'].str.split('|').explode().value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']

fig_genre = px.bar(genre_counts.head(10), x='Genre', y='Count', color='Genre', title="Top 10 Most Rated Genres")
st.plotly_chart(fig_genre, use_container_width=True)

# Average Rating per Genre
st.subheader("⭐ Average Rating by Genre")

genre_ratings = df.copy()
genre_ratings = genre_ratings.dropna(subset=['genres'])
genre_ratings['genre'] = genre_ratings['genres'].str.split('|')
genre_ratings = genre_ratings.explode('genre')
avg_rating_by_genre = genre_ratings.groupby('genre')['rating'].mean().sort_values(ascending=False).reset_index()

fig_avg = px.bar(avg_rating_by_genre, x='genre', y='rating', title="Average Rating per Genre", color='genre')
st.plotly_chart(fig_avg, use_container_width=True)

# Rating Distribution
st.subheader("📊 Distribution of Ratings")

fig_hist = px.histogram(df, x='rating', nbins=20, title='Ratings Histogram')
st.plotly_chart(fig_hist, use_container_width=True)

# Word Cloud of Tags
st.subheader("🏷️ Word Cloud from Tags")

if df['tag'].notna().sum() > 0:
    text = ' '.join(df['tag'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
else:
    st.warning("No tag data available to generate word cloud.")

# Correlation Matrix
st.subheader("🔍 Correlation Between Numeric Features")

numeric_df = df[['userId', 'movieId', 'rating']]
corr = numeric_df.corr()
fig_corr, ax_corr = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)

# Interactive: Top Movies
st.subheader("🏆 Top Rated Movies")

min_votes = st.slider("Minimum number of ratings:", 10, 100, 50)

top_movies = (
    df.groupby('title')
    .agg(avg_rating=('rating', 'mean'), num_ratings=('rating', 'count'))
    .query("num_ratings >= @min_votes")
    .sort_values(by='avg_rating', ascending=False)
    .head(10)
    .reset_index()
)

st.dataframe(top_movies)

# Footer
st.markdown("""
---
✅ This dashboard provides a clear summary of movie rating behavior.
🧠 This dataset is ideal for **recommendation systems** in online retail, as it contains:
- User-item interaction (`userId`, `movieId`, `rating`)
- Categorical info for content-based filtering (`genres`, `tags`)
- Time-based behavior (`timestamp_x`)
""")


# In[ ]:




