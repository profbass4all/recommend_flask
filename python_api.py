# python_api.py
from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
import sys
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


def get_recommendations_from_loaded_data(movie_title, movie_collection, combined_similarity, indices, top_n=15):
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
    print("Loading data and building model...", file=sys.stderr)

    url = "https://drive.google.com/file/d/1PEjFZiaD67GsWbVGzfr2uktDp3K6zLxq/view?usp=sharing"
    gdd = read_gd(url)

    if gdd is None:
        print("Failed to load data from Google Drive.", file=sys.stderr)
        return 

    try:
        df = pd.read_csv(gdd)
        movie_collection = df 

        
        for i in range(1, 5):
            star_col = f'Star{i}'
            if star_col not in movie_collection.columns:
                movie_collection[star_col] = '' 

        movie_collection['Certificate'] = movie_collection['Certificate'].fillna('Unrated')
        movie_collection['Stars'] = movie_collection['Star1'].fillna('') + '|' + movie_collection['Star2'].fillna('') + "|" + movie_collection['Star3'].fillna('') + "|" + movie_collection['Star4'].fillna('')

        text_features=['Overview']
        categorical_features=['Director', 'Stars', 'Genre','Certificate']
        weights={
            "Genre": 5,
            "Director": 3,
            "Certificate": 2,
            'Stars': 4.5,
            'Overview': 4,
            'Series_Title': 2 
        }
        n_components=100

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
                else:
                    print(f"Warning: Feature '{feature}' resulted in an empty vocabulary. Skipping.", file=sys.stderr)


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


        combined_similarity = categorical_features_similarity + text_features_similarity

        if combined_similarity.size > 0 and np.max(combined_similarity) > np.min(combined_similarity):
            scaler = MinMaxScaler()
            combined_similarity = scaler.fit_transform(combined_similarity)
        elif combined_similarity.size > 0:
             combined_similarity = np.ones_like(combined_similarity) * (combined_similarity[0,0] > 0)
        else:
            print("Warning: Combined similarity matrix is empty.", file=sys.stderr)
            combined_similarity = np.zeros((len(movie_collection), len(movie_collection)))


        indices = pd.Series(movie_collection.index, index = movie_collection['Series_Title']) 
        print("Data and model loaded successfully.", file=sys.stderr)

    except Exception as e:
        print(f'An error occurred during data loading and model building: {e}', file=sys.stderr)
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