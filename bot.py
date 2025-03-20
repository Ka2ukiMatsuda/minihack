import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from utils.aws.s3 import upload_image_to_s3
from utils.aws.dynamodb import put_item_to_dynamodb
from utils.openai import summarize_image
from utils.embed import encode
from utils.azure import analyze_image

from commands.whiteboard import whiteboard_command
from events.message import message_events

# AWS
import boto3


# Azure Computer Vision
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

# OpenAI
from openai import OpenAI


def check_environment_variables(name: str):
    try:
        os.environ[name]
    except KeyError:
        print(f"Missing environment variable '{name}''")
        print("Set them before running.")
        exit()


REQUIRED_ENVS = [
    "VISION_ENDPOINT",
    "VISION_KEY",
    "SLACK_BOT_TOKEN",
    "SLACK_SIGNING_SECRET",
    "SLACK_APP_TOKEN",
]

ENVS = {}

for env in REQUIRED_ENVS:
    check_environment_variables(env)
    ENVS[env] = os.environ[env]


SUMMARY_TABLE_NAME = "SummaryTable"
OCR_TABLE_NAME = "OcrTable"
BUCKET_NAME = "minihack-whiteboard-images"


openai_client = OpenAI()
azure_client = ImageAnalysisClient(
    endpoint=ENVS["VISION_ENDPOINT"], credential=AzureKeyCredential(ENVS["VISION_KEY"])
)

s3_client = boto3.client("s3")

dynamodb_client = boto3.resource("dynamodb")
summary_table = dynamodb_client.Table(SUMMARY_TABLE_NAME)  # type: ignore
ocr_table = dynamodb_client.Table(OCR_TABLE_NAME)  # type: ignore

slack_app = App(
    token=ENVS["SLACK_BOT_TOKEN"],
    signing_secret=ENVS["SLACK_SIGNING_SECRET"],
    name="VisualSummarizer2.0",
)


@slack_app.command("/whiteboard")
def handle_whiteboard_command(ack, say, client, command):
    whiteboard_command(
        ack, say, client, command, summary_table, ocr_table, s3_client, encode
    )


@slack_app.event("message")
def handle_message_events(body, say, client):
    message_events(
        body,
        say,
        slack_bot_token=ENVS["SLACK_BOT_TOKEN"],
        ocr_table=ocr_table,
        summary_table=summary_table,
        s3_client=s3_client,
        openai_client=openai_client,
        azure_client=azure_client,
        encode=encode,
        summarize_image=summarize_image,
        analyze_image=analyze_image,
        put_item_to_dynamodb=put_item_to_dynamodb,
        upload_image_to_s3=upload_image_to_s3,
        bucket_name=BUCKET_NAME,
    )


def main():
    handler = SocketModeHandler(slack_app, ENVS["SLACK_APP_TOKEN"])
    handler.start()


if __name__ == "__main__":
    main()
