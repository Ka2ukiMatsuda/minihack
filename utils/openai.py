import base64


def encode_byte_to_base64(byte_data):
    encoded_string = base64.b64encode(byte_data).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def get_prompt(context):
    return f"ホワイトボードの内容を図表を含め説明してください．以下はこのホワイトボードの画像に対する補足情報です．補足情報を用いてホワイトボードの内容を説明してください．# 補足情報: {context}"


def summarize_image(client, image_byte_data, context):
    image_base64 = encode_byte_to_base64(image_byte_data)
    # OpenAI APIを使用して要約
    response = client.chat.completions.create(
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
