from azure.ai.vision.imageanalysis.models import VisualFeatures


def analyze_image(client, image_data: bytes):
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
    )
    text = ""
    if result.read is not None:
        for line in result.read.blocks[0].lines:
            text += "".join([word.text for word in line.words])
    return text
