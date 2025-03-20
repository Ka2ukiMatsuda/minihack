import boto3

import torch.nn.functional as F
import torch

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import json


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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
ocrs = []
id_to_image_name = {}
for image_name in data:
    llm_summary = data[image_name]["chatgpt_summary"]
    id_to_image_name[len(input_texts)] = image_name
    input_texts.append(llm_summary)
    ocrs.append(data[image_name]["ocr_text"])


embeddings = encode(input_texts)

# DynamoDBのテーブル名
SUMMARY_TABLE_NAME = "SummaryTable"
OCR_TABLE_NAME = "OcrTable"

# DynamoDBクライアントの作成
dynamodb = boto3.resource("dynamodb")
ocrTable = dynamodb.Table(OCR_TABLE_NAME)
summaryTable = dynamodb.Table(SUMMARY_TABLE_NAME)


def put_item_to_dynamodb(item, table):
    """
    DynamoDBにアイテムを追加する関数
    :param item: 辞書型のアイテム
    """
    try:
        response = table.put_item(Item=item)
        print("PutItem succeeded:", response)
    except Exception as e:
        print("Error putting item:", e)


for i in range(len(input_texts)):
    id = i
    image_name = id_to_image_name[i].split(".")[0]
    extension = id_to_image_name[i].split(".")[1]
    summary_content = str(embeddings[i].tolist())
    ocr_content = ocrs[i]

    put_item_to_dynamodb(
        {
            "id": str(id),
            "image_name": image_name,
            "extension": extension,
            "content": summary_content,
        },
        summaryTable,
    )

    put_item_to_dynamodb(
        {
            "id": str(id),
            "image_name": image_name,
            "extension": extension,
            "content": ocr_content,
        },
        ocrTable,
    )
