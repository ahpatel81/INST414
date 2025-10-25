import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_colwidth", 200)

COLUMNS = ["movie_title", "director_name", "genres", "imdb_score"]
df = pd.read_csv("movie_metadata.csv", na_filter=False)[COLUMNS].head(500).reset_index(drop=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("|", " ").replace(",", " ").strip()
    return " ".join(text.split())

df["director_name"] = df["director_name"].apply(clean_text)
df["genres"] = df["genres"].apply(clean_text)
df["imdb_score"] = pd.to_numeric(df["imdb_score"], errors="coerce")

df["combined_features"] = (df["genres"] + " " + df["director_name"]).str.strip()

vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
similarity_matrix = cosine_similarity(tfidf_matrix)

title_to_index = {
    title.strip().lower(): i
    for i, title in enumerate(df["movie_title"])
    if isinstance(title, str) and title.strip()
}

def get_similar_movies(title, top_n=10):
    key = title.strip().lower()
    if key not in title_to_index:
        return pd.DataFrame()
    idx = title_to_index[key]
    scores = similarity_matrix[idx]
    top_matches = scores.argsort()[::-1]
    top_matches = [i for i in top_matches if i != idx][:top_n]
    return pd.DataFrame({
        "Rank": range(1, len(top_matches) + 1),
        "Movie": df.loc[top_matches, "movie_title"].values,
        "Similarity": scores[top_matches].round(4),
        "Genres": df.loc[top_matches, "genres"].values,
        "Director": df.loc[top_matches, "director_name"].values,
        "IMDB Score": df.loc[top_matches, "imdb_score"].values
    })

print(f"\nTop 10 similar movies to Spider-Man 3") 
print(get_similar_movies("Spider-Man 3")) 

print(f"\nTop 10 similar movies to Harry Potter and the Half-Blood Prince") 
print(get_similar_movies("Harry Potter and the Half-Blood Prince")) 

print(f"\nTop 10 similar movies to Monsters University") 
print(get_similar_movies("Monsters University"))
