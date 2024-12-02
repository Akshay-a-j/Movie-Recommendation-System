# Movie-Recommendation-System
This project implements a movie recommendation system using Locality-Sensitive Hashing (LSH) technique, providing fast and efficient similarity-based movie recommendations. The system uses the MovieLens dataset and features a clean Streamlit interface for easy interaction.

## Overview
The system combines genre and title features to find similar movies using LSH, which allows for quick approximate nearest neighbor search. It provides recommendations through two methods:
  1. Search by movie title
  2. Direct movie ID input

## Technical Details
The recommendation system uses:
  1. Locality-Sensitive Hashing (LSH) for efficient similarity search
  2. TF-IDF vectorization for movie titles
  3. One-hot encoding for movie genres
  4. Multiple hash tables and projections for better accuracy
  5. Adaptive thresholding for optimal bucket distribution
  6. Neighboring bucket search for improved recall

## Requirements
First, ensure you have Python 3.8+ installed on your system. Then install the required packages:
