from torch import Tensor
from transformers import AutoTokenizer

import json

import onnxruntime as ort
import numpy as np


def normalize(embeddings):
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)  # L2ノルムを計算
    return embeddings / (norms + 1e-8)  # ゼロ除算防止のための微小値を加える


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# モデルのパス
onnx_path = "onnx_model/multilingual-e5-large.onnx"

# ONNX Runtime の推論セッションを作成
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# トークナイザーのロード
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_embedding(text: str):
    # `query: ` や `passage: ` をつけるのが E5 の仕様
    formatted_text = text

    # 入力データの準備
    inputs = tokenizer(formatted_text, return_tensors="np")

    # ONNX 用に変換
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # 推論の実行
    ort_outs = session.run(None, ort_inputs)

    # last_hidden_state の平均をベクトルとして使用
    embedding = np.mean(ort_outs[0], axis=1).tolist()

    return embedding


with open("./extracted_data.json", "r") as f:
    data = json.load(f)


input_texts: list[str] = []
id_to_image_name = {}
for image_name in data:
    llm_summary = data[image_name]["chatgpt_summary"]
    id_to_image_name[len(input_texts)] = image_name
    input_texts.append(llm_summary)


embeddings = []
for text in input_texts:
    embeddings.append(get_embedding(text))
embeddings = np.array(embeddings)  # type: ignore
embeddings = normalize(embeddings.reshape(embeddings.shape[0], -1))


print(embeddings)
print(embeddings.shape)
