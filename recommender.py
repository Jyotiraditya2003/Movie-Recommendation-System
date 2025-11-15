import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[['title', 'genres', 'director', 'cast']].dropna()
        self.df['combined_features'] = self.df.apply(self.combine_features, axis=1)

        vectorizer = CountVectorizer(stop_words='english')
        self.feature_vectors = vectorizer.fit_transform(self.df['combined_features'])
        self.similarity = cosine_similarity(self.feature_vectors)

    def combine_features(self, row):
        return f"{row['genres']} {row['director']} {row['cast']}"

    def recommend(self, movie_title, n=5):
        if movie_title not in self.df['title'].values:
            return "Movie not found in dataset."
        
        idx = self.df[self.df['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(self.similarity[idx]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        recommended = [self.df.iloc[i[0]]['title'] for i in sorted_scores]
        return recommended
