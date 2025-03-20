import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# モデルのパス
onnx_path = "onnx_model/multilingual-e5-large.onnx"

# ONNX Runtime の推論セッションを作成
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# トークナイザーのロード
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_embedding(text: str):
    # `query: ` や `passage: ` をつけるのが E5 の仕様
    formatted_text = f"query: {text}"

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


# 推論テスト
text = "What is the capital of France?"
embedding = get_embedding(text)
print("埋め込みベクトル:", np.array(embedding).shape)
