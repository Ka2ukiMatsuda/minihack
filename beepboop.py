from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
import os

import boto3
from io import BytesIO

s3 = boto3.client("s3")

app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
    name="beepboop",
)


@app.event("app_mention")
def send_image_url(body, say, client):
    try:
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


def main():
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()


if __name__ == "__main__":
    main()
