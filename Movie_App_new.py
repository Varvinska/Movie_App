#!/usr/bin/env python
# coding: utf-8

# In[12]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎬 Movie Insight Dashboard", layout="wide")
st.title("🎬 Movie Insight Dashboard")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Varvinska/Movie_App/refs/heads/main/movie_dataset.csv'
    return pd.read_csv(url)

df = load_data()


# Define genres list
all_genres = sorted(set(g for sublist in df['genres'].dropna().str.split('|') for g in sublist))

# Sidebar filters
selected_genres = st.sidebar.multiselect("Select Genre(s)", options=all_genres, default=all_genres[:3])
rating_range = st.sidebar.slider("Select Rating Range", 0.0, 5.0, (3.0, 5.0), 0.5)
available_tags = sorted(df['tag'].dropna().unique())
selected_tags = st.sidebar.multiselect("Select Tag(s)", options=available_tags)


# Get all unique non-null tags
all_tags = sorted(df["tag"].dropna().unique())
selected_tags = st.sidebar.multiselect("Select Tag(s)", options=all_tags)

# --- Data Filtering ---
filtered_df = df.copy()

# Filter by selected genres
if selected_genres:
    df = df[df["genres"].isin(genres)]
    
# Filter by rating
filtered_df = filtered_df[(filtered_df['rating'] >= rating_range[0]) & (filtered_df['rating'] <= rating_range[1])]

# Filter by selected tags
if selected_tags:
    filtered_df = filtered_df[filtered_df['tag'].isin(selected_tags)]

# --- Show Raw Data ---
if st.checkbox("Show raw data"):
    st.dataframe(filtered_df.head(20))

# --- Genre Analysis ---
st.subheader("🍿 Most Rated Genres")
genre_counts = filtered_df['genres'].str.split('|').explode().value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']
fig_genre = px.bar(genre_counts.head(10), x='Genre', y='Count', color='Genre', title="Top 10 Most Rated Genres")
st.plotly_chart(fig_genre, use_container_width=True)

# --- Average Rating by Genre ---
st.subheader("⭐ Average Rating by Genre")
genre_ratings = filtered_df.dropna(subset=['genres']).copy()
genre_ratings['genre'] = genre_ratings['genres'].str.split('|')
genre_ratings = genre_ratings.explode('genre')
avg_rating_by_genre = genre_ratings.groupby('genre')['rating'].mean().sort_values(ascending=False).reset_index()
fig_avg = px.bar(avg_rating_by_genre, x='genre', y='rating', title="Average Rating per Genre", color='genre')
st.plotly_chart(fig_avg, use_container_width=True)

# --- Rating Distribution ---
st.subheader("📊 Distribution of Ratings")
fig_hist = px.histogram(filtered_df, x='rating', nbins=20, title='Ratings Histogram')
st.plotly_chart(fig_hist, use_container_width=True)

# --- Word Cloud ---
st.subheader("🏷️ Word Cloud from Tags")
if filtered_df['tag'].notna().sum() > 0:
    text = ' '.join(filtered_df['tag'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
else:
    st.warning("No tag data available to generate word cloud.")

# --- Correlation Matrix ---
st.subheader("🔍 Correlation Between Numeric Features")
numeric_df = filtered_df[['userId', 'movieId', 'rating']]
if not numeric_df.empty:
    corr = numeric_df.corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
else:
    st.info("Not enough numeric data after filtering.")

# --- Top Movies ---
st.subheader("🏆 Top Rated Movies")
min_votes = st.slider("Minimum number of ratings:", 10, 100, 50)
top_movies = (
    filtered_df.groupby('title')
    .agg(avg_rating=('rating', 'mean'), num_ratings=('rating', 'count'))
    .query("num_ratings >= @min_votes")
    .sort_values(by='avg_rating', ascending=False)
    .head(10)
    .reset_index()
)

st.dataframe(top_movies)

# --- Footer ---
st.markdown("""
---
✅ This dashboard provides a clear summary of movie rating behavior.
🧠 Ideal for **recommendation systems**, it uses:
- User-item interactions
- Genre and tag content
- Rating distributions
""")


# In[ ]:




