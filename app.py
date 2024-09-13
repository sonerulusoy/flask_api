from flask import Flask, request, jsonify
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

word2vec_model = Word2Vec.load(
    "C:/Users/uluso/Desktop/data-and-model/word2vec_model_trained.model"
)

train_df = pd.read_csv("C:/Users/uluso/Desktop/data-and-model/train_data.csv")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def clean_and_tokenize(text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    lemmatizer = nltk.WordNetLemmatizer()
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    return cleaned_tokens


def get_product_vector(tokens, word2vec_model):
    valid_tokens = [token for token in tokens if token in word2vec_model.wv]
    if valid_tokens:
        return np.mean([word2vec_model.wv[token] for token in valid_tokens], axis=0)
    else:
        return None


def get_search_vector(query_tokens, word2vec_model):  
    return get_product_vector(query_tokens, word2vec_model)

    query_tokens = clean_and_tokenize(search_query)
    search_vector = get_search_vector(query_tokens, word2vec_model)

    if search_vector is None:
        return []

    similarities = []


def find_similar_products(search_query, word2vec_model, top_n=5):
    query_tokens = clean_and_tokenize(search_query)
    search_vector = get_search_vector(query_tokens, word2vec_model)

    if search_vector is None:
        return []

    similarities = []

    for i, row in train_df.iterrows():
        product_tokens = clean_and_tokenize(
            row["name"]
            + " "
            + row["author"]
            + " "
            + row["category"]
            + " "
            + row["description"]
        )
        product_vector = get_product_vector(product_tokens, word2vec_model)

        if product_vector is not None:
            similarity = cosine_similarity([search_vector], [product_vector])[0][0]
            similarities.append(
                {
                    "name": row["name"],
                    "category": row["category"],
                    "image": row.get("image", ""),  
                    "price": row.get("price", "N/A"),
                    "similarity": similarity,  
                }
            )

    similar_products = sorted(
        similarities, key=lambda x: x["similarity"], reverse=True
    )[:top_n]
    return similar_products


# Ana sayfa endpoint'i
@app.route("/", methods=["GET"])
def home():
    return "Ürün öneri API na hoşgeldiniz.!"


# @app.route("/recommend", methods=["POST"])
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    product_description = data.get("description")

    if not product_description:
        return jsonify({"error": "Urun açıklaması yok."}), 400

    similar_products = find_similar_products(product_description, word2vec_model)

    if not similar_products:
        return jsonify({"error": "Benzer urun bulunamadı."}), 404

    return jsonify(
        {
            "description": product_description,
            "recommendations": [
                {
                    "name": product["name"],
                    "category": product["category"],
                    "image": product["image"],
                    "price": product["price"],
                    "similarity": float(product["similarity"])  
                }
                for product in similar_products
            ],
        }
    )



if __name__ == "__main__":
    app.run(debug=True)
