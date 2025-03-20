from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

import os
import re

import boto3
from io import BytesIO

import torch.nn.functional as F
import torch
import faiss

from torch import Tensor

from transformers import AutoTokenizer, AutoModel

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
SUMMARY_TABLE_NAME = "SummaryTable"
OCR_TABLE_NAME = "OcrTable"

app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
    name="beepboop",
)

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


@app.event("app_mention")
def send_image_url(body, say, client, event):
    # try:
    ocr_table = dynamodb.Table(OCR_TABLE_NAME)  # type: ignore
    summary_table = dynamodb.Table(SUMMARY_TABLE_NAME)  # type: ignore
    summary_info = summary_table.scan()["Items"]
    ocr_info = ocr_table.scan()["Items"]
    summary_info.sort(key=lambda x: x["id"])
    ocr_info.sort(key=lambda x: x["id"])
    embeddings = torch.tensor([eval(x["content"]) for x in summary_info])
    ocrs = [x["content"] for x in ocr_info]

    mention = body["event"]
    channel_id = mention["channel"]

    print(len(embeddings))

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.detach().numpy())  # type: ignore

    query = event["text"]
    query = re.sub(r"<@.*?>", "", query)
    print(query)
    query_embedding = encode(query)

    k = len(summary_info)
    D, I = index.search(query_embedding, k)  # type: ignore

    actual_results = []
    for idx, score in zip(I[0], D[0]):
        if query in ocrs[idx]:
            score += 0.1
        actual_results.append((idx, score))
        print(idx, score)

    actual_results.sort(key=lambda x: x[1], reverse=True)
    for idx, similarity in actual_results:
        print(f"{idx}:{summary_info[idx]['image_name']}:{similarity}")

    uploaded_files = []
    top_k = 3
    for idx, similarity in actual_results[:top_k]:
        # S3から画像をバイナリデータとして取得
        bucket_name = "minihack-whiteboard-images"
        object_key = (
            f"{summary_info[idx]['image_name']}.{summary_info[idx]['extension']}"
        )
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response["Body"].read()
        image_file = BytesIO(image_data)

        result = client.files_upload_v2(
            channel=channel_id,
            file=image_file,
            filename=object_key,
            title="result",
            thread_ts=mention["ts"],
        )

        uploaded_files.append(result["file"]["id"])

    client.chat_postMessage(
        channel=channel_id,
        thread_ts=mention["ts"],
        text="こちらでどうですか？",
    )
    # except SlackApiError as e:
    #     print(f"Error occurred in Slack API: {e.response['error']}")
    # except Exception as e:
    #     print(f"Error occurred: {e}")


def main():
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()


if __name__ == "__main__":
    main()
