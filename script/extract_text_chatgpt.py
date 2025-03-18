from openai import OpenAI
import os
import base64
# OpenAIクライアントをインスタンス化
client = OpenAI()

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

# imageフォルダ内の全てのjpgファイルに対して処理を行う
image_folder = 'image'
for file_name in os.listdir(image_folder):
    if file_name.endswith('.jpg'):
        file_path = os.path.join(image_folder, file_name)

        image_base64 = encode_image_to_base64(file_path)

        # ユーザからのコンテキスト情報を受け取る
        print(f"{file_name}に関するコンテキストを入力してください（例: これはXXの時の画像です）: ")
        context = input()


        # OpenAI APIを使用してOCRを実行
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"ホワイトボードの内容を図表を含め説明してください．以下はこのホワイトボードの画像に対する補足情報です．補足情報を用いてホワイトボードの内容を説明してください．# 補足情報: {context}"},
                        {"type": "image_url",
                         "image_url": {
                             "url": image_base64,
                             "detail": "high"
                         }
                        }
                    ]
                }
            ]
        )

        output_text = response.choices[0].message.content
        print(f"Text for {file_name}:")
        print(output_text)
        print("--------------------------------")

        import json

        with open('extracted_data.json', 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)

        extracted_data[file_name]['context'] = context
        extracted_data[file_name]['chatgpt_summary'] = output_text

        with open('extracted_data.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
