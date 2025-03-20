import json
import logging
import os

from slack_bolt import Ack, App, Respond
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import boto3
import numpy as np

from io import BytesIO

from torch import Tensor
from transformers import AutoTokenizer

import json

import onnxruntime as ort
import re


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


# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger()
logger.setLevel(logging.INFO)


# def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# def average_pool(last_hidden_states, attention_mask):
#     # Convert PyTorch tensors to NumPy arrays
#     last_hidden_states_np = last_hidden_states.cpu().numpy()
#     attention_mask_np = attention_mask.cpu().numpy()
#
#     # Perform masking operation
#     last_hidden_np = np.where(
#         attention_mask_np[..., None].astype(bool), last_hidden_states_np, 0.0
#     )
#
#     # Compute sum and average
#     sum_hidden = np.sum(last_hidden_np, axis=1)
#     sum_attention = np.sum(attention_mask_np, axis=1)[..., None]
#     average_hidden = sum_hidden / sum_attention
#
#     return average_hidden


# tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
# model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")


def encode(inputs):
    # Tokenize the input texts
    # batch_dict = tokenizer(
    #     inputs, max_length=512, padding=True, truncation=True, return_tensors="pt"
    # )

    # with torch.no_grad():
    #     outputs = model(**batch_dict)
    # embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # return embeddings
    pass


app = App(
    process_before_response=True,
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)
s3 = boto3.client("s3")

dynamodb = boto3.resource("dynamodb")

SUMMARY_TABLE_NAME = "SummaryTable"


@app.event("app_mention")
def send_image_url(body, say, client, event):
    try:
        summary_table = dynamodb.Table(SUMMARY_TABLE_NAME)  # type: ignore
        response = summary_table.scan()

        logger.info(f"response: {response}")

        embeddings = []
        text = event["text"]
        text = re.sub(r"<@.*?>", "", text)
        embeddings.append(get_embedding(text))
        embeddings = np.array(embeddings)  # type: ignore
        embeddings = normalize(embeddings.reshape(embeddings.shape[0], -1))

        print(embeddings)
        print(embeddings.shape)

        return

        embeddings = np.array([])

        bucket_name = "minihack-whiteboard-images"
        object_key = "IMG_1960.jpg"

        # S3から画像をバイナリデータとして取得
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response["Body"].read()
        image_file = BytesIO(image_data)

        mention = body["event"]
        channel_id = mention["channel"]

        print(body)
        print(mention)
        print(channel_id)

        result = client.files_upload_v2(
            channel=channel_id,
            file=image_file,
            filename=object_key,
            title="S3から取得した画像",
            initial_comment="こちらですか？",
        )
    except SlackApiError as e:
        print(f"Error occurred in Slack API: {e.response['error']}")
    except Exception as e:
        print(f"Error occurred: {e}")


def lambda_handler(event, context):
    logger.info(f"event:\n{event}")
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
