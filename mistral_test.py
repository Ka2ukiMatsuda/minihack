from mistralai import Mistral, DocumentURLChunk
from pathlib import Path
import json
import os
import base64
from time import sleep


def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


def list_files(directory):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def exec_mistral(image_path):
    # Getting the base64 string
    # base64_image = encode_image(image_path)

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    pdf_file = Path(image_path)
    assert pdf_file.is_file()

    # PDFファイルをアップロード
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr",
    )

    # アップロードしたファイルに対して、署名付きURLを取得
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    # PDFファイルをOCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png",
        },
    )

    # ocr_response = client.ocr.process(
    #     model="mistral-ocr-latest",
    #     document={
    #         "type": "image_url",
    #         "image_url": f"data:image/jpeg;base64,{base64_image}",
    #     },
    # )

    response_dict = json.loads(ocr_response.model_dump_json())
    json_string = json.dumps(response_dict, indent=2, ensure_ascii=False)
    # OCR結果をファイルに保存。
    result_file = Path(f"./result/{image_path.split('/')[2].split('.')[0]}.json")
    result_file.write_text(json_string, encoding="utf-8")


# 使用例
files = list_files("./image")

# for file in files:
#     # print(file)
#     exec_mistral(f"./image/{file}")
#     sleep(2)
#     print(f"OCR処理が完了しました。{file}")
exec_mistral("./image/IMG_7508.jpg")
