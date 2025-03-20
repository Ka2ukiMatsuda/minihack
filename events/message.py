import requests


def message_events(
    body,
    say,
    slack_bot_token,
    ocr_table,
    summary_table,
    s3_client,
    openai_client,
    azure_client,
    encode,
    summarize_image,
    analyze_image,
    put_item_to_dynamodb,
    upload_image_to_s3,
    bucket_name,
):
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
    summary_text = summarize_image(openai_client, image_buffer, text)
    summary_embedding_str = str(encode(summary_text)[0].tolist())
    ocr_text = analyze_image(azure_client, image_buffer)

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

    upload_image_to_s3(
        s3_client, bucket_name, image_buffer, f"{image_name}.{extension}"
    )

    say("画像をデータベースに登録しました！")
