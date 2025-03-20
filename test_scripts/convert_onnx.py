import torch
from transformers import AutoModel, AutoTokenizer
import onnx
from pathlib import Path

# モデル名
model_name = "intfloat/multilingual-e5-large"

# トークナイザーとモデルをロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ダミー入力の作成
dummy_text = "query: What is the capital of France?"
dummy_inputs = tokenizer(dummy_text, return_tensors="pt")

# ONNX の保存先
onnx_dir = Path("onnx_model")
onnx_dir.mkdir(exist_ok=True)
onnx_path = onnx_dir / "multilingual-e5-large.onnx"

# モデルを ONNX にエクスポート
torch.onnx.export(
    model,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    str(onnx_path),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"},
    },
    opset_version=14,
)

print(f"ONNXモデルが {onnx_path} に保存されました")
