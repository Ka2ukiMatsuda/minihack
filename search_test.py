import torch.nn.functional as F
import torch
import faiss

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import json


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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


with open("./extracted_data.json", "r") as f:
    data = json.load(f)


input_texts: list[str] = []
id_to_image_name = {}
for image_name in data:
    llm_summary = data[image_name]["chatgpt_summary"]
    id_to_image_name[len(input_texts)] = image_name
    input_texts.append(llm_summary)


embeddings = encode(input_texts)


print(embeddings)
print(embeddings.shape)


index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings.detach().numpy())  # type: ignore

query = input("Enter a query: ")

query_embedding = encode(query)

k = 11

D, I = index.search(query_embedding, k)  # type: ignore

for i in range(k):
    print(f"Rank {i + 1}")
    print(f"Similarity: {D[0][i]}")
    print(f"Sentence: {id_to_image_name[I[0][i]]}")
    print()
