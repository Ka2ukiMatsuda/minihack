from sentence_transformers import SentenceTransformer, util
import faiss

# モデルのロード
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# 比較する文章
sentences = [
    "自然言語処理には多くの挑戦がありますが、面白いです。",
    "自然言語は面白くないな",
    "好きな食べ物は何ですか?",
    "どこにお住まいですか?",
    "朝の電車は混みますね",
    "今日は良いお天気ですね",
    "最近景気悪いですね",
]

# 文章をベクトルに変換
embeddings = model.encode(
    sentences=sentences, convert_to_numpy=True, normalize_embeddings=True
)

print(embeddings.shape)

print(embeddings.sum(axis=1))

embedding_dim = 768
index = faiss.IndexFlatIP(embedding_dim)

index.add(embeddings)

query = "自然言語処理は非常に興味深い分野です。"

query_embedding = model.encode(
    query, convert_to_numpy=True, normalize_embeddings=True
).reshape(1, -1)

k = 10

D, I = index.search(query_embedding, k)

for i in range(k):
    print(f"Rank {i + 1}")
    print(f"Similarity: {D[0][i]}")
    print(f"Sentence: {sentences[I[0][i]]}")
    print()
