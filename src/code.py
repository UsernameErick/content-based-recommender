import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer # tfidf
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz # fuzzy matching (нечеткое совпадение, чтобы можно было вводить запрос с ошибкой)
import re

# для того, чтобы файл искался по абсолютному пути, а не в папке со скриптом
base_dir = os.path.dirname(__file__) # папка src. __file__ путь к файлу
dataset_path = os.path.join(base_dir, "..", "data", "steam.csv") # join умно(смотрит какая операционка на компе) соединяет все пути в один 
df = pd.read_csv(dataset_path)

df = df[['name', 'genres', 'steamspy_tags']]

df['tags'] = df['genres'].fillna("") + " " + df['steamspy_tags'].fillna("")
df['tags'] = df['tags'].str.lower()

# tfidf
vectorizer = TfidfVectorizer(stop_words='english') # стоп-слова the is and of ...

tfidf_matrix = vectorizer.fit_transform(df['tags']) # строка - название игры, столбцы - теги. создаются векторы

#cosine similarity. измеряет угол между векторами tf-idf. чем угол меньше(cos0 = 1 > одинаковые игры), тем более похожи.
# similarity = cosine_similarity(tfidf_matrix) # (насколько близки игры в кластере друг к другу). убираем эту строку, так как для полной матрицы похожести будет нужно 81 гб оперативы

df['clean_name'] = (df['name'].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)) # создается колонка без TM, R и прочих знаков. очищенная строка
# индексация игр
indices = pd.Series(df.index, index=df['clean_name'].str.lower()) # создан словарь название игры : индекс игры ("индекс : название" было бы медленнее)
# функция рекомендации
def recommend_games(game_name, n=5):
    game_name = find_game_fastbetter(game_name) # поиск полного имени игры и присвоение его к game_name
    if game_name is None:
        print("| Game not found! |")
        return
    print(f"Found: {game_name}") # вывод самой близкой игры которую система нашла (вероятнее всего эту игру мы и ввели)
    idx = indices[game_name.lower()] # idx будет равняться индексу игры которую ввели (из словаря "название : индекс")
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix) # считаем схожесть вектора одной нашей игры со всеми векторами игр
    sim_scores = list(enumerate(sim_scores[0])) # enumerate сделает из каждого элемента матрицы похожести пару из индекса и элемента. [0, 0.91, 0.88] => (0, 0), (1, 0.91), (2, 0.88) 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse = True) # сортировка по x[1] то есть по признаку похожести, а после reverse - (1, 0.91), (2, 0.88), (0, 0). игра с индексом 1 похожа на 0.91, а игра с индексом 0 вообще не похожа на нашу indices[game_name]
    sim_scores = sim_scores[1:n+1] # у первой строки будет похожесть 1 - это сама игра. надо ее убрать
    game_indices = [i[0] for i in sim_scores]
    print("Recommended games for", game_name)
    print(df['name'].iloc[game_indices])

def find_game(query): # fuzz matching
    query = re.sub(r'[^a-zA-Z0-9 ]', '', query.lower()) # просто нормализация текста, чтобы поиск был успешнее, и убирает всякие TM и другие знаки
    match = process.extractOne(query, df['clean_name'], scorer=fuzz.token_sort_ratio, score_cutoff=60) # поиск по названию. score_cutoff не выводит варианты ниже 60 похожести
    if match is None:
        return None
    return match[0] # вернет только название (0 название, 1 уровень похожести, 2 индекс игры)

def find_game_fastbetter(query):
    query = query.lower()
    matches = df[df["clean_name"].str.contains(query, na=False)] # поиск всех игр, где встречается запрошенное название query
    if len(matches) > 0:
        return matches["clean_name"].iloc[0]
    # если не сработало идем дальше
    query_words = set(query.split()) # разбиваем название игры на слова - {"assassin", "creed"}
    best_score = 0
    best_game = None
    for name in df["clean_name"]: # substring match
        score = len(query_words & set(name.split())) # сравнивается сколько слов пересеклось(&) с name в clean names. {"assassin","creed"} & {"assassins","creed","unity"} → {"creed"}. score = 1  
        if score > best_score:
            best_score = score
            best_game = name
    return best_game

print("=================================================") 
#recommend_games("Zuma's Revenge")
#print(df[df['name'].str.contains("star wars", case=False, na=False)])

# для запуска через проводник
if __name__ == "__main__":
    while True:
        query = input("\nEnter your game name or 'exit': ")
        if query.lower() == "exit":
            break
        recommend_games(query)