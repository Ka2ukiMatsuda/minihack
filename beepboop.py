import os
import requests
import base64
from io import BytesIO
from time import sleep

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

# AWS
import boto3

import torch.nn.functional as F
import torch
from torch import Tensor

from transformers import AutoTokenizer, AutoModel

import faiss

# Azure Computer Vision
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# OpenAI
from openai import OpenAI


# Set the values of your computer vision endpoint and computer vision key
# as environment variables:


def check_environment_variables(name: str):
    try:
        val = os.environ[name]
        return val
    except KeyError:
        print(f"Missing environment variable '{name}''")
        print("Set them before running.")
        exit()


vision_endpoint = check_environment_variables("VISION_ENDPOINT")
vision_key = check_environment_variables("VISION_KEY")
slack_bot_token = check_environment_variables("SLACK_BOT_TOKEN")
slack_signing_secret = check_environment_variables("SLACK_SIGNING_SECRET")
slack_app_token = check_environment_variables("SLACK_APP_TOKEN")


SUMMARY_TABLE_NAME = "SummaryTable"
OCR_TABLE_NAME = "OcrTable"
BUCKET_NAME = "minihack-whiteboard-images"

TOP_K = 3

azure_client = ImageAnalysisClient(
    endpoint=vision_endpoint, credential=AzureKeyCredential(vision_key)
)
s3_client = boto3.client("s3")
dynamodb_client = boto3.resource("dynamodb")
summary_table = dynamodb_client.Table(SUMMARY_TABLE_NAME)  # type: ignore
ocr_table = dynamodb_client.Table(OCR_TABLE_NAME)  # type: ignore

openai_client = OpenAI()
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
embedding_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

slack_app = App(
    token=slack_bot_token,
    signing_secret=slack_signing_secret,
    name="beepboop",
)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode(inputs):
    # Tokenize the input texts
    batch_dict = tokenizer(
        inputs, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = embedding_model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


@slack_app.command("/whiteboard")
def handle_whiteboard_command(ack, say, client, command):
    ack()

    # try:
    # SummaryTableからデータを取得
    summary_info = summary_table.scan()["Items"]
    summary_info.sort(key=lambda x: x["id"])
    embeddings = torch.tensor([eval(x["content"]) for x in summary_info])
    # OCRTableからデータを取得
    ocr_info = ocr_table.scan()["Items"]
    ocr_info.sort(key=lambda x: x["id"])
    ocrs = [x["content"].lower() for x in ocr_info]

    channel_id = command["channel_id"]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.detach().numpy())  # type: ignore

    query = command["text"]
    print("Query: ", query)

    response = client.chat_postMessage(
        channel=channel_id,
        text=f"検索クエリ：{query}",
    )
    query_embedding = encode(query)

    k = len(summary_info)
    D, I = index.search(query_embedding, k)  # type: ignore

    actual_results = []
    for idx, score in zip(I[0], D[0]):
        count = 0
        query_words = query.split()
        for word in query_words:
            if query in ocrs[idx]:
                count += 1
        actual_score = score + 0.1 * count / len(query_words)
        actual_results.append((idx, actual_score))

    actual_results.sort(key=lambda x: x[1], reverse=True)
    for idx, similarity in actual_results:
        print(f"{idx}:{summary_info[idx]['image_name']}:{similarity}")

    thread_ts = response["ts"]

    uploaded_files = []
    for i, (idx, similarity) in enumerate(actual_results[:TOP_K]):
        # get image binary data from s3
        object_key = (
            f"{summary_info[idx]['image_name']}.{summary_info[idx]['extension']}"
        )
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_key)
        image_data = response["Body"].read()
        image_file = BytesIO(image_data)

        result = client.files_upload_v2(
            channel=channel_id,
            file=image_file,
            filename=object_key,
            title=f"No.{i + 1}",
            thread_ts=thread_ts,
            initial_comment=f"No.{i + 1}",
        )
        sleep(0.2)

        uploaded_files.append(result["file"]["id"])

    # except SlackApiError as e:
    #     print(f"Error occurred in Slack API: {e.response['error']}")
    # except Exception as e:
    #     print(f"Error occurred: {e}")


def encode_byte_to_base64(byte_data):
    encoded_string = base64.b64encode(byte_data).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def get_prompt(context):
    return f"ホワイトボードの内容を図表を含め説明してください．以下はこのホワイトボードの画像に対する補足情報です．補足情報を用いてホワイトボードの内容を説明してください．# 補足情報: {context}"


def summarize_image(byte_data, context):
    image_base64 = encode_byte_to_base64(byte_data)
    # OpenAI APIを使用して要約
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": get_prompt(context),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64, "detail": "high"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def analyze_image(image_data: bytes):
    result = azure_client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
    )
    text = ""
    if result.read is not None:
        for line in result.read.blocks[0].lines:
            text += "".join([word.text for word in line.words])
    return text


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


def upload_image_to_s3(image_data, image_name):
    s3_client.put_object(Bucket=BUCKET_NAME, Key=image_name, Body=image_data)


@slack_app.event("message")
def handle_message_events(body, say, client):
    event = body.get("event", {})
    text = event.get("text", "")  # メッセージのテキストを取得
    files = event.get("files", [])  # 添付されたファイル情報

    images = []
    for file in files:
        if file["mimetype"].startswith("image/"):  # 画像ファイルのみ処理
            image_url = file["url_private"]

            # Download the image
            headers = {"Authorization": f"Bearer {slack_bot_token}"}
            response = requests.get(image_url, headers=headers)

            if response.status_code == 200:
                id = file["id"]
                extension = file["name"].split(".")[-1]
                image_name = f"{id}.{extension}"
                image_buffer = response.content
                images.append((image_name, image_buffer, extension))

    if not images:
        return

    if len(images) >= 2:
        say("画像は1枚だけにしてください！")
        return

    image_name, image_buffer, extension = images[0]

    ocr_info = ocr_table.scan()["Items"]
    max_id = max([int(x["id"]) for x in ocr_info])
    summary_text = summarize_image(image_buffer, text)
    summary_embedding_str = str(encode(summary_text)[0].tolist())
    ocr_text = analyze_image(image_buffer)

    put_item_to_dynamodb(
        {
            "id": str(max_id + 1),
            "image_name": image_name,
            "extension": extension,
            "content": summary_embedding_str,
        },
        summary_table,
    )

    put_item_to_dynamodb(
        {
            "id": str(max_id + 1),
            "image_name": image_name,
            "extension": extension,
            "content": ocr_text,
        },
        ocr_table,
    )

    upload_image_to_s3(image_buffer, f"{image_name}.{extension}")

    say("画像をデータベースに登録しました！")


def main():
    handler = SocketModeHandler(slack_app, slack_app_token)
    handler.start()


if __name__ == "__main__":
    main()
