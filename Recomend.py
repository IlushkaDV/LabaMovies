import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
import warnings

warnings.filterwarnings("ignore")

ratings = pd.read_csv("ratings_small.csv")
movies = pd.read_csv("movies_metadata.csv", low_memory=False)

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
X = user_movie_matrix.values
movie_ids = user_movie_matrix.columns

X_binary = (X > 0).astype(float)

print("Обучение RBM модели...")
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=20, verbose=True, random_state=42)
rbm.fit(X_binary)

reconstructed = rbm.gibbs(X_binary)

while True:
    print("\nДоступные ID пользователей:", list(user_movie_matrix.index))
    user_input = input("Введите ID пользователя для рекомендаций (или 'q' для выхода): ")

    if user_input.lower() == 'q':
        break

    try:
        user_id = int(user_input)
        if user_id not in user_movie_matrix.index:
            print(f"Ошибка: пользователь с ID {user_id} не найден.")
            continue

        user_index = np.where(user_movie_matrix.index == user_id)[0][0]
        user_ratings = X[user_index]
        user_reconstructed = reconstructed[user_index]

        unseen = np.where(user_ratings == 0)[0]
        recommended_indices = unseen[np.argsort(user_reconstructed[unseen])[::-1]]  # Сортировка по убыванию

        movies_clean = movies.dropna(subset=['id', 'title'])
        movies_clean['id'] = movies_clean['id'].astype(str)

        recommended_titles = []
        for movie_idx in recommended_indices:
            movie_id = str(movie_ids[movie_idx])
            movie_title = movies_clean[movies_clean['id'] == movie_id]['title']
            if not movie_title.empty:
                recommended_titles.append(movie_title.values[0])
            if len(recommended_titles) >= 5:  # Останавливаемся, когда набрали 5 фильмов
                break

        print(f"\nРекомендованные фильмы для пользователя {user_id}:\n")
        if recommended_titles:
            for i, title in enumerate(recommended_titles, 1):
                print(f"{i}. {title}")
        else:
            print("Не удалось найти рекомендации. Возможно, данные о фильмах отсутствуют.")

    except ValueError:
        print("Ошибка: введите корректный числовой ID пользователя.")