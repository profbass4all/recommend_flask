# python_api.py
import sys

print("Python script started!", file=sys.stderr)

from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import requests 

app = Flask(__name__)

movie_collection = None
combined_similarity = None
indices = None

def read_gd(sharingurl):
    try:
        file_id = sharingurl.split('/')[-2]
        download_url = 'https://drive.google.com/uc?export=download&id=' + file_id
        
        response = requests.get(download_url)
        response.raise_for_status() 
        csv_raw = StringIO(response.text)
        return csv_raw
    except Exception as e:
        print(f"Error reading Google Drive file: {e}", file=sys.stderr)
        return None 


def get_recommendations_from_loaded_data(movie_title, movie_collection, combined_similarity, indices, top_n=6):
    if movie_collection is None or combined_similarity is None or indices is None:
        print("Error: Data or model not loaded.", file=sys.stderr)
        return [] 
    
    if movie_title not in indices:
        print(f"Error: Movie '{movie_title}' not found in the collection.", file=sys.stderr)
        return [] 

    try:
        index_of_movie = indices[movie_title]

        similarity_scores = list(enumerate(combined_similarity[index_of_movie]))
        similarity_scores = sorted(similarity_scores, key = lambda x: x[1], reverse = True)

        recommended_indices = [score[0] for score in similarity_scores if score[0] != index_of_movie and score[1] > 0]

        
        output_columns = ['Series_Title', 'Overview', 'Director', 'Stars', 'Genre', 'Certificate', 'Released_Year', 'IMDB_Rating', 'Runtime']
        output_columns = [col for col in output_columns if col in movie_collection.columns]


        
        recommended_movies_df = movie_collection.iloc[recommended_indices]
        final_output_df = recommended_movies_df[output_columns].head(top_n)

        
        final_output_df = final_output_df.where(pd.notnull(final_output_df), None)

        return final_output_df.to_dict(orient='records')

    except Exception as e:
        print(f'An error occurred during recommendation lookup: {e}', file=sys.stderr)
        return [] 

def load_data_and_build_model():
    global movie_collection, combined_similarity, indices
    print("Starting data loading and model building...", file=sys.stderr)
    sys.stderr.flush() # Explicitly flush output

    # Update with your actual Google Drive URL
    url = "https://drive.google.com/file/d/1PEjFZiaD67GsWbVGzfr2uktDp3K6zLxq/view?usp=sharing"

    try:
        print(f"Attempting to read data from Google Drive URL: {url}", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        gdd = read_gd(url)

        if gdd is None:
            print("Failed to load data from Google Drive (read_gd returned None).", file=sys.stderr)
            sys.stderr.flush() # Explicitly flush output
            return # Exit if data loading failed

        print("Successfully read data stream from Google Drive.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output

        try:
            print("Attempting to read CSV into DataFrame...", file=sys.stderr)
            sys.stderr.flush() # Explicitly flush output
            df = pd.read_csv(gdd)
            movie_collection = df # Store the loaded DataFrame globally
            print(f"Successfully loaded {len(df)} rows into DataFrame.", file=sys.stderr)
            sys.stderr.flush() # Explicitly flush output

        except Exception as e:
             print(f"Error reading CSV into DataFrame: {e}", file=sys.stderr)
             sys.stderr.flush() # Explicitly flush output
             # Set global variables to None to indicate failure
             movie_collection = None
             combined_similarity = None
             indices = None
             return # Exit if CSV reading failed


        # --- Data Preprocessing (Same as before) ---
        print("Starting data preprocessing...", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # ... (rest of preprocessing code) ...
        print("Finished data preprocessing.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output


        # TF-IDF and SVD for Text Features
        print("Starting TF-IDF and SVD for text features...", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # ... (rest of TF-IDF/SVD code) ...
        print("Finished TF-IDF and SVD.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output


        # TF-IDF for Categorical Features
        print("Starting TF-IDF for categorical features...", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # ... (rest of categorical TF-IDF code) ...
        print("Finished categorical TF-IDF.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output

        print("Calculating combined similarity...", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # ... (rest of similarity calculation code) ...
        print("Finished combined similarity.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output


        # Normalize similarity scores
        print("Normalizing similarity scores...", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # ... (rest of normalization code) ...
        print("Finished normalization.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output


        print("Creating movie index...", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # ... (rest of index creation code) ...
        print("Finished creating index.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output


        print("Data and model loaded successfully.", file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output

    except Exception as e:
        print(f'An error occurred during data loading and model building: {e}', file=sys.stderr)
        sys.stderr.flush() # Explicitly flush output
        # Set global variables to None to indicate failure
        movie_collection = None
        combined_similarity = None
        indices = None


@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title') 

    if not movie_title:
        return jsonify({"error": "Missing movie title query parameter"}), 400

    recommendations = get_recommendations_from_loaded_data(
        movie_title,
        movie_collection,
        combined_similarity,
        indices
    )

    if recommendations is None: 
        return jsonify({"error": "Internal server error during recommendation lookup"}), 500

    if not recommendations: 
        if indices is not None and movie_title not in indices:
            return jsonify({"error": f"Movie '{movie_title}' not found"}), 404
        else:
            return jsonify({"message": f"No recommendations found for '{movie_title}'"}), 200


    return jsonify(recommendations), 200


@app.route('/movies', methods=['GET'])
def get_all_movies():
    if movie_collection is None:
        return jsonify({"error": "Movie data not loaded"}), 500

    try:
        all_movies_columns = ['Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']
        all_movies_columns = [col for col in all_movies_columns if col in movie_collection.columns]

        all_movies_data = movie_collection[all_movies_columns].where(pd.notnull(movie_collection[all_movies_columns]), None).to_dict(orient='records')
        return jsonify(all_movies_data), 200
    except Exception as e:
        print(f"Error getting all movies: {e}", file=sys.stderr)
        return jsonify({"error": "Internal server error getting all movies"}), 500


if __name__ == '__main__':
    load_data_and_build_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000) 