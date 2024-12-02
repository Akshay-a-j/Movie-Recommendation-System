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
1. Clone the repository:
```
git clone https://github.com/yourusername/movie-recommendation-lsh.git
cd movie-recommendation-lsh
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit application:
```
streamlit run app.py
```
The application will be available at http://localhost:8501 in your web browser.

## Using the System

1. Choose your input method:
   * Search by title: Type a movie name and select from matching results
   * Enter Movie ID: Directly input a MovieLens movie ID
2. Adjust the number of recommendations using the slider
3. Click "Get Recommendations" to see similar movies

## System Parameters
The system uses the following default parameters:

  1. Number of Hash Tables: 5
  2. Number of Projections: 50
  3. Feature Dimensions: 300 (for TF-IDF)

These parameters provide a good balance between accuracy and performance for the MovieLens Small dataset.

## Features

1. Fast similarity search using LSH
2. Intuitive user interface
3. Dual search methods (title or ID)
4. Adjustable number of recommendations
5. Genre and title-based similarity
6. Fallback recommendations when needed

## Algorithm Details
The system implements LSH with the following key features:

1. Feature Creation:
   * Combines genre one-hot encoding with TF-IDF title features
   * Normalizes feature vectors for consistent distance calculations

3. LSH Implementation:
   * Uses multiple hash tables for better recall
   * Implements bit-flipping for neighboring bucket search
   * Uses adaptive thresholding for optimal bucket distribution

4. Recommendation Generation:
   * Collects candidates from multiple hash tables
   * Computes exact distances for final ranking
   * Provides genre-based fallback recommendations




