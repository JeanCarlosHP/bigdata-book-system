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
    try:
        book_index = books.index[books['bookID'] == book_id].tolist()[0]

        similarity_scores = list(enumerate(similarity_matrix[book_index]))

        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommended_indices = [idx for idx, score in sorted_scores[1:num_recommendations+1]]

        recommended_books = books.iloc[recommended_indices]
        return recommended_books['bookID'].tolist()
    except IndexError:
        return []

train_data = books.sample(frac=0.8, random_state=42)
test_data = books.drop(train_data.index)

test_data['relevant_books'] = test_data.apply(
    lambda row: train_data[train_data['authors'] == row['authors']]['bookID'].tolist(),
    axis=1
)

def evaluate_recommendations(test_data, num_recommendations=5):
    total_relevant = 0
    total_recommended = 0
    total_relevant_recommended = 0

    for _, row in test_data.iterrows():
        book_id = row['bookID']
        relevant_books = set(row['relevant_books'])

        if not relevant_books:
            continue

        recommendations = set(recommend_books_by_features(book_id, num_recommendations))
        relevant_recommended = recommendations & relevant_books

        total_relevant += len(relevant_books)
        total_recommended += len(recommendations)
        total_relevant_recommended += len(relevant_recommended)

    precision = total_relevant_recommended / total_recommended if total_recommended > 0 else 0
    recall = total_relevant_recommended / total_relevant if total_relevant > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

precision, recall, f1 = evaluate_recommendations(test_data, num_recommendations=5)
print(f"Precisão: {precision:.2f}, Revocação: {recall:.2f}, F1-Score: {f1:.2f}")
