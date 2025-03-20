import torch
import faiss
from io import BytesIO
from time import sleep
from typing import cast

TOP_K = 3
BUCKET_NAME = "minihack-whiteboard-images"


def whiteboard_command(
    ack, say, client, command, summary_table, ocr_table, s3_client, encode
):
    ack()

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

    query: str = cast(str, command["text"])
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
            if word.lower() in ocrs[idx]:
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
        sleep(0.5)

        uploaded_files.append(result["file"]["id"])
