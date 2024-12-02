import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('books.csv')

books = data[['bookID', 'title', 'authors', 'language_code', 'publisher']].dropna()

books['features'] = books['authors'] + " " + books['language_code'] + " " + books['publisher']

vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(books['features'])

similarity_matrix = cosine_similarity(feature_vectors, feature_vectors)

def recommend_books_by_features(book_id, num_recommendations=5):
    book_index = books.index[books['bookID'] == book_id].tolist()[0]

    similarity_scores = list(enumerate(similarity_matrix[book_index]))

    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [idx for idx, score in sorted_scores[1:num_recommendations+1]]

    recommended_books = books.iloc[recommended_indices]
    return recommended_books[['bookID', 'title', 'authors', 'language_code', 'publisher']]

book_to_recommend = 500
print(f"Recomendações para o livro com ID {book_to_recommend}:")
print(recommend_books_by_features(book_to_recommend))
