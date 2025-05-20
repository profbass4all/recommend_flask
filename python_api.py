# python_api.py
import sys
print("Python script started!", file=sys.stderr)
sys.stderr.flush()

from flask import Flask, request, jsonify
import pandas as pd
# Removed: from io import StringIO # No longer needed for local file
# Removed: import requests # No longer needed for Google Drive download

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import os # Import os to help with file paths

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global variables to store data and model (loaded once on startup) ---
movie_collection = None
combined_similarity = None
indices = None

# --- Define the local CSV file path ---
# Assuming the CSV file is in the same directory as python_api.py
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'imdb_top_1000.csv') # Replace 'imdb_top_1000.csv' with the actual filename

# --- Removed: read_gd function is no longer needed ---
# def read_gd(sharingurl):
#     ...

# --- Modified Recommendation Logic (Callable function) ---
# This function remains mostly the same, it uses the global variables
def get_recommendations_from_loaded_data(movie_title, movie_collection, combined_similarity, indices, top_n=15):
    if movie_collection is None or combined_similarity is None or indices is None:
        print("Error: Data or model not loaded.", file=sys.stderr)
        sys.stderr.flush()
        return [] # Return empty list if data isn't ready

    if movie_title not in indices:
        print(f"Error: Movie '{movie_title}' not found in the collection.", file=sys.stderr)
        sys.stderr.flush()
        return [] # Return empty list if movie not found

    try:
        index_of_movie = indices[movie_title]

        similarity_scores = list(enumerate(combined_similarity[index_of_movie]))
        similarity_scores = sorted(similarity_scores, key = lambda x: x[1], reverse = True)

        # Get recommended indices, excluding the movie itself and scores <= 0
        recommended_indices = [score[0] for score in similarity_scores if score[0] != index_of_movie and score[1] > 0]

        # Define output columns - ensure these match the columns expected by your frontend
        # This list should be consistent with what's available in your movie_collection DataFrame
        output_columns = ['Series_Title', 'Overview', 'Director', 'Stars', 'Genre', 'Certificate', 'Released_Year', 'IMDB_Rating', 'Runtime']
        # Filter to only include columns that exist in the DataFrame
        output_columns = [col for col in output_columns if col in movie_collection.columns]


        # Return the recommended movies as a list of dictionaries (JSON format)
        recommended_movies_df = movie_collection.iloc[recommended_indices]
        final_output_df = recommended_movies_df[output_columns].head(top_n)

        # Handle potential NaN values in the output DataFrame before converting to dict
        # Convert NaN to None, which translates to null in JSON
        final_output_df = final_output_df.where(pd.notnull(final_output_df), None)

        return final_output_df.to_dict(orient='records')

    except Exception as e:
        print(f'An error occurred during recommendation lookup: {e}', file=sys.stderr)
        sys.stderr.flush()
        return [] # Return empty list on error


# --- Data Loading and Model Building (Runs once on startup) ---
def load_data_and_build_model():
    global movie_collection, combined_similarity, indices
    print("Starting data loading and model building...", file=sys.stderr)
    sys.stderr.flush()

    # Removed: Google Drive URL and read_gd call
    # url = "https://drive.google.com/file/d/1PEjFZiaD67GsWbVGzfr2uktDp3K6zLxq/view?usp=sharing"
    # gdd = read_gd(url)

    try:
        # --- Read directly from the local CSV file ---
        print(f"Attempting to read CSV from local file: {CSV_FILE_PATH}", file=sys.stderr)
        sys.stderr.flush()

        # Check if the file exists
        if not os.path.exists(CSV_FILE_PATH):
             print(f"Error: Local CSV file not found at {CSV_FILE_PATH}", file=sys.stderr)
             sys.stderr.flush()
             # Set global variables to None to indicate failure
             movie_collection = None
             combined_similarity = None
             indices = None
             return # Exit if file not found


        df = pd.read_csv(CSV_FILE_PATH) # Read directly from the local file path
        movie_collection = df # Store the loaded DataFrame globally
        print(f"Successfully loaded {len(df)} rows into DataFrame from local file.", file=sys.stderr)
        sys.stderr.flush()


        # --- Data Preprocessing (Same as before) ---
        print("Starting data preprocessing...", file=sys.stderr)
        sys.stderr.flush()
        # Prepare columns (handling potential missing Star columns)
        for i in range(1, 5):
            star_col = f'Star{i}'
            if star_col not in movie_collection.columns:
                 movie_collection[star_col] = '' # Add missing star columns as empty strings

        movie_collection['Certificate'] = movie_collection['Certificate'].fillna('Unrated')
        movie_collection['Stars'] = movie_collection['Star1'].fillna('') + '|' + movie_collection['Star2'].fillna('') + "|" + movie_collection['Star3'].fillna('') + "|" + movie_collection['Star4'].fillna('')

        text_features=['Overview'] # Corrected text features
        categorical_features=['Director', 'Stars', 'Genre','Certificate']
        weights={
            "Genre": 5,
            "Director": 3,
            "Certificate": 2,
            'Stars': 4.5,
            'Overview': 4,
            'Series_Title': 2 # Weight for Series_Title if used in text features
        }
        n_components=100

        # Data Preprocessing for Categorical Features (only if they exist)
        for feature in categorical_features:
            if feature in movie_collection.columns:
                movie_collection[feature] = movie_collection[feature].astype(str).apply(
                    lambda x: (
                        [item.strip() for item in re.split(r'\|', x) if item.strip()]
                        if isinstance(x, str) and str(x).strip()
                        else (
                            x if isinstance(x, list)
                            else []
                        ))
                )
        print("Finished data preprocessing.", file=sys.stderr)
        sys.stderr.flush()


        # TF-IDF and SVD for Text Features
        print("Starting TF-IDF and SVD for text features...", file=sys.stderr)
        sys.stderr.flush()
        text_features_similarity = np.zeros((len(movie_collection), len(movie_collection)))
        vectorizer = TfidfVectorizer(stop_words='english')

        for feature in text_features:
             if feature in movie_collection.columns:
                feature_data = movie_collection[feature].fillna('').astype(str)
                vectorizer_matrix = vectorizer.fit_transform(feature_data)
                if vectorizer_matrix.shape[1] > 0:
                    actual_n_components = min(n_components, vectorizer_matrix.shape[1], vectorizer_matrix.shape[0])
                    if actual_n_components > 0:
                         svd = TruncatedSVD(n_components=actual_n_components, random_state=42)
                         reduced_vectors = svd.fit_transform(vectorizer_matrix)
                         weight = weights.get(feature, 1) if weights else 1
                         text_features_similarity += weight * cosine_similarity(reduced_vectors)
                    else:
                         print(f"Warning: Cannot perform SVD on feature '{feature}'. Skipping.", file=sys.stderr)
                         sys.stderr.flush()
                else:
                    print(f"Warning: Feature '{feature}' resulted in an empty vocabulary. Skipping.", file=sys.stderr)
                    sys.stderr.flush()
        print("Finished TF-IDF and SVD.", file=sys.stderr)
        sys.stderr.flush()


        # TF-IDF for Categorical Features
        print("Starting TF-IDF for categorical features...", file=sys.stderr)
        sys.stderr.flush()
        categorical_features_similarity = np.zeros((len(movie_collection), len(movie_collection)))
        vectorizer_cat = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)

        for feature in categorical_features:
            if feature in movie_collection.columns:
                feature_data = movie_collection[feature].apply(lambda x: [str(item) for item in x] if isinstance(x, list) else [])
                if feature_data.sum():
                     vectorizer_matrix = vectorizer_cat.fit_transform(feature_data)
                     weight = weights.get(feature, 1) if weights else 1
                     categorical_features_similarity += weight * cosine_similarity(vectorizer_matrix)
                else:
                     print(f"Warning: Feature '{feature}' resulted in empty data. Skipping.", file=sys.stderr)
                     sys.stderr.flush()
        print("Finished categorical TF-IDF.", file=sys.stderr)
        sys.stderr.flush()

        print("Calculating combined similarity...", file=sys.stderr)
        sys.stderr.flush()
        combined_similarity = categorical_features_similarity + text_features_similarity
        print("Finished combined similarity.", file=sys.stderr)
        sys.stderr.flush()


        # Normalize similarity scores
        print("Normalizing similarity scores...", file=sys.stderr)
        sys.stderr.flush()
        if combined_similarity.size > 0 and np.max(combined_similarity) > np.min(combined_similarity):
             scaler = MinMaxScaler()
             combined_similarity = scaler.fit_transform(combined_similarity)
        elif combined_similarity.size > 0:
             combined_similarity = np.ones_like(combined_similarity) * (combined_similarity[0,0] > 0)
        else:
             print("Warning: Combined similarity matrix is empty. Setting to zeros.", file=sys.stderr)
             sys.stderr.flush()
             combined_similarity = np.zeros((len(movie_collection), len(movie_collection))) # Initialize as zeros if empty
        print("Finished normalization.", file=sys.stderr)
        sys.stderr.flush()


        print("Creating movie index...", file=sys.stderr)
        sys.stderr.flush()
        indices = pd.Series(movie_collection.index, index = movie_collection['Series_Title']) # Store indices globally
        print("Finished creating index.", file=sys.stderr)
        sys.stderr.flush()


        print("Data and model loaded successfully.", file=sys.stderr)
        sys.stderr.flush()

    except Exception as e:
        print(f'An error occurred during data loading and model building: {e}', file=sys.stderr)
        sys.stderr.flush()
        # Set global variables to None to indicate failure
        movie_collection = None
        combined_similarity = None
        indices = None


# ... (rest of your Flask endpoints and startup logic) ...

# --- Startup Logic ---
if __name__ == '__main__':
    # Load data and build model when the script starts
    load_data_and_build_model()
    # Run the Flask development server
    # In production, use a WSGI server like Gunicorn or uWSGI
    # Make sure the port is set correctly for Render
    # Use os.environ.get('PORT', 5000) to get the port assigned by Render
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) # Set debug=False for production
