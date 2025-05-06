import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
from scipy.sparse import csr_matrix

ratings = pd.read_csv('ratings_small.csv')
movies = pd.read_csv('movies_metadata.csv', low_memory=False)

ratings['rating'] = np.where(ratings['rating'] >= 4, 1, 0)

user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

min_movie_ratings = 10
min_user_ratings = 5
filtered_users = user_movie_matrix.sum(axis=1) >= min_user_ratings
filtered_movies = user_movie_matrix.sum(axis=0) >= min_movie_ratings
user_movie_matrix = user_movie_matrix.loc[filtered_users, filtered_movies]

if user_movie_matrix.shape[0] == 0 or user_movie_matrix.shape[1] == 0:
    raise ValueError("Недостаточно данных после фильтрации.")

sparse_matrix = csr_matrix(user_movie_matrix.values)

rbm = BernoulliRBM(
    n_components=100,
    learning_rate=0.05,
    n_iter=20,
    batch_size=32,
    random_state=42
)
rbm.fit(sparse_matrix)

def recommend_movies(user_id, top_n=5, movies_df=movies):
    if user_id not in user_movie_matrix.index:
        available_users = user_movie_matrix.index.tolist()
        print(f"Пользователь {user_id} не найден. Доступные пользователи: {available_users[:10]}...")
        return []

    user_idx = user_movie_matrix.index.get_loc(user_id)
    user_ratings = sparse_matrix[user_idx].toarray()
    predicted_scores = rbm.gibbs(user_ratings).flatten()

    unrated_mask = (user_ratings.flatten() == 0)
    unrated_movie_ids = user_movie_matrix.columns[unrated_mask]
    unrated_scores = predicted_scores[unrated_mask]

    top_indices = np.argsort(unrated_scores)[::-1][:top_n]
    top_movie_ids = unrated_movie_ids[top_indices]

    movies_df['id'] = movies_df['id'].astype(str)
    top_movie_ids = top_movie_ids.astype(str)
    recommendations = movies_df[movies_df['id'].isin(top_movie_ids)]['title'].tolist()

    return recommendations if recommendations else ["Не удалось найти фильмы для рекомендаций"]

available_users = user_movie_matrix.index.tolist()
user_id = available_users[0]
print("\nДоступные пользователи (первые 10):", user_movie_matrix.index.tolist()[:20])

while True:
    try:
        user_id = int(input("Введите ID пользователя из списка выше: "))
        if user_id in user_movie_matrix.index:
            break
        else:
            print(f"Ошибка: пользователь {user_id} не найден. Попробуйте еще раз.")
    except ValueError:
        print("Ошибка: введите числовой ID.")

print(f"Рекомендации для пользователя {user_id}:")
recommended = recommend_movies(user_id)
for i, title in enumerate(recommended, 1):
    print(f"{i}. {title}")