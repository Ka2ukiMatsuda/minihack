from sentence_transformers import SentenceTransformer, util

# モデルのロード
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer("intfloat/multilingual-e5-large")

# 比較する文章
sentence1 = "自然言語処理は非常に興味深い分野です。"
sentence2 = "自然言語処理には多くの挑戦がありますが、面白いです。"
sentence3 = "自然言語処理面白くないな"

# 文章をベクトルに変換
embeddings = model.encode(
    sentences=[sentence1, sentence2, sentence3], convert_to_numpy=True
)

# コサイン類似度の計算
cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0]
cosine_score2 = util.pytorch_cos_sim(embeddings[0], embeddings[2])[0][0]

print(f"文章1と文章2の類似度: {cosine_score}")
print(f"文章1と文章3の類似度: {cosine_score2}")

print(embeddings.shape)
