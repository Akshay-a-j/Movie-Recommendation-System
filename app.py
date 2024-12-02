import streamlit as st
from main import MovieRecommender
st.title("Movie Recommendation System")

@st.cache_resource
def load_recommender():
    recommender = MovieRecommender()
    recommender.fit('movies.csv')
    return recommender

recommender = load_recommender()

# Initialize movie_id as None
movie_id = None

# Input options
search_method = st.radio("Select Input Method", ["Search by Title", "Enter Movie ID"])

if search_method == "Search by Title":
    search_query = st.text_input("Search for a movie:")
    if search_query:
        matches = recommender.movies_df[
            recommender.movies_df['title'].str.contains(search_query, case=False)
        ]
        if not matches.empty:
            selected_title = st.selectbox("Select a movie:", matches['title'].tolist())
            movie_id = matches[matches['title'] == selected_title]['movieId'].iloc[0]
        else:
            st.warning("No movies found matching your search.")
else:
    movie_id = st.number_input("Enter Movie ID:", min_value=1, value=1)

if movie_id is not None:
    k = st.slider("Number of recommendations:", 1, 10, 5)
    
    if st.button("Get Recommendations"):
        # Query movie details
        query_movie = recommender.movies_df[recommender.movies_df['movieId'] == movie_id].iloc[0]
        st.subheader("Query Movie:")
        st.write(f"Title: {query_movie['title']}")
        st.write(f"Genres: {query_movie['genres']}")
        
        # Get and display recommendations
        st.subheader("Recommendations:")
        similar_indices = recommender.find_similar_movies(movie_id, k)
        for idx in similar_indices:
            movie = recommender.movies_df.iloc[idx]
            st.write(f"• {movie['title']} ({movie['genres']})")

# Add sidebar information
with st.sidebar:
    st.subheader("About")
    st.write("""
    This system uses Locality-Sensitive Hashing (LSH) to quickly find similar movies 
    based on their genres and titles. The recommendations are based on both genre 
    similarity and title features.
    """)
    
    st.subheader("System Parameters")
    st.write(f"Number of Hash Tables: {recommender.n_tables}")
    st.write(f"Number of Projections: {recommender.n_projections}")
