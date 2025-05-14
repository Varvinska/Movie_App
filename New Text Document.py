import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Movie Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("movie_dataset.csv")

data = load_data()

st.title("Interactive Movie Dashboard")
st.markdown("Explore movies by genre, ratings, and tags — tailored for ages 18–35.")

# Sidebar filters
genres = st.sidebar.multiselect("Select Genre(s)", options=sorted(data["genres"].unique()), default=["Action", "Comedy", "Drama"])
rating_range = st.sidebar.slider("Select Rating Range", 0.0, 5.0, (3.0, 5.0), 0.5)
tags = st.sidebar.multiselect("Select Tag(s)", options=sorted(data["tag"].dropna().unique()))

# Filter data
filtered_data = data[
    (data["genres"].isin(genres)) &
    (data["rating"] >= rating_range[0]) &
    (data["rating"] <= rating_range[1])
]

if tags:
    filtered_data = filtered_data[filtered_data["tag"].isin(tags)]

# Tabs for visualizations
tab1, tab2, tab3 = st.tabs(["Genre Overview", "Tags", "Insights"])

with tab1:
    st.subheader("Average Rating per Genre")
    genre_rating = filtered_data.groupby("genres")["rating"].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(data=genre_rating, x="genres", y="rating", palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    st.markdown("Genres with the highest average ratings are most appreciated by young adults.")

with tab2:
    st.subheader("Popular Tags Word Cloud")
    tags_text = " ".join(filtered_data["tag"].dropna().astype(str))
    if tags_text:
        wordcloud = WordCloud(background_color="white", width=800, height=300).generate(tags_text)
        plt.figure(figsize=(10, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())
    else:
        st.info("No tags available for selected filters.")

with tab3:
    st.subheader("Number of Ratings per Genre")
    genre_counts = filtered_data.groupby("genres")["rating"].count().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(data=genre_counts, x="genres", y="rating", palette="coolwarm")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    st.markdown("Useful for identifying popular genres for targeted recommendations.")