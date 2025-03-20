import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from io import BytesIO

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Set image_url to the URL of an image that you want to analyze.
image_path = "./image/IMG_7202.jpg"
# Read the image into a byte array
# image_data = open(image_path, "rb").read()
with open(image_path, "rb") as f:
    input_bytes = f.read()

# リサイズ & 前処理
image_data = input_bytes


# Get a caption for the image. This will be a synchronously (blocking) call.
result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.READ],
)

print("Image analysis results:")

# Print text (OCR) analysis results to the console
print(" Read:")
if result.read is not None:
    text = ""
    for line in result.read.blocks[0].lines:
        # text += "".join([word.text for word in line.words])
        for word in line.words:
            print(word.text)
    print(text)
