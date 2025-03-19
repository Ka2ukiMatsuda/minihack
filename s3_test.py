import boto3
from io import BytesIO
from PIL import Image

s3 = boto3.client("s3")
bucket_name = "minihack-whiteboard-images"
object_key = "IMG_1960.jpg"

# S3から画像をバイナリデータとして取得
response = s3.get_object(Bucket=bucket_name, Key=object_key)
image_data = response["Body"].read()

# Pillow (PIL) を使って画像を開く
image = Image.open(BytesIO(image_data))
image.show()  # 画像を開く
