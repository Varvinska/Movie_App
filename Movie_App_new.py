#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("movie_dataset.csv")

# Page config
st.set_page_config(page_title="Movie Insights Dashboard", layout="wide")

# Header
st.title("Movie Dashboard for Young Adults (18–35)")
st.markdown("Explore trends and data insights suitable for predictive modeling in online retail.")

# Sidebar filters
st.sidebar.header("Filter Options")
genres = df['genres'].dropna().unique().tolist()
selected_genres = st.sidebar.multiselect("Select Genre(s)", genres, default=genres)
rating_range = st.sidebar.slider("Select Rating Range", float(df['rating'].min()), float(df['rating'].max()), (2.0, 10.0))

# Filtered Data
filtered_df = df[
    (df['genres'].isin(selected_genres)) &
    (df['rating'] >= rating_range[0]) &
    (df['rating'] <= rating_range[1])
]

# Summary Metrics
st.subheader("Summary Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Rating", round(filtered_df["rating"].mean(), 2))
with col2:
    st.metric("Most Common Genre", filtered_df["genres"].mode()[0])
with col3:
    st.metric("Top Release Year", int(filtered_df["timestamp_x"].mode()[0]))

# Genre popularity
st.subheader("Genre Popularity")
genre_count = filtered_df['genres'].value_counts().head(10)
fig1, ax1 = plt.subplots()
sns.barplot(x=genre_count.values, y=genre_count.index, palette='cool', ax=ax1)
ax1.set_xlabel("Number of Movies")
st.pyplot(fig1)

# Rating distribution
st.subheader("Ratings Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['rating'], bins=20, kde=True, ax=ax2, color='coral')
st.pyplot(fig2)

# Ratings over time
st.subheader("Rating Trends Over Time")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=filtered_df, x='timestamp_x', y='rating', hue='genres', alpha=0.6, ax=ax3)
st.pyplot(fig3)

# ML Suitability section
st.sidebar.markdown("ML Suitability Insights")
st.sidebar.markdown("""
- Multiple numerical & categorical features.
- Predictable target variable (Rating, Success).
- Useful for:
  - Movie recommendation.
  - Revenue prediction.
  - Genre-based targeting.
""")

# Correlation matrix
st.subheader("Feature Correlation Matrix")
numeric_df = filtered_df.select_dtypes(include='number')
if not numeric_df.empty:
    fig4, ax4 = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)
else:
    st.info("Not enough numeric data for correlation matrix.")


# In[ ]:




