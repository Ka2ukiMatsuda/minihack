import torch.nn.functional as F
import torch
import faiss

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = [
    "自然言語処理には多くの挑戦がありますが、面白いです。",
    "自然言語は面白くないな",
    "好きな食べ物は何ですか?",
    "どこにお住まいですか?",
    "朝の電車は混みますね",
    "今日は良いお天気ですね",
    "最近景気悪いですね",
]

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")


def encode(inputs):
    # Tokenize the input texts
    batch_dict = tokenizer(
        inputs, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


embeddings = encode(input_texts)


print(embeddings)
print(embeddings.shape)


index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings.detach().numpy())  # type: ignore

query = "自然言語処理は非常に興味深い分野です。"
# query = "NLPが好きです"

query_embedding = encode(query)

k = 10

D, I = index.search(query_embedding, k)  # type: ignore

for i in range(k):
    print(f"Rank {i + 1}")
    print(f"Similarity: {D[0][i]}")
    print(f"Sentence: {input_texts[I[0][i]]}")
    print()
