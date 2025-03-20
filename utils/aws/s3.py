def upload_image_to_s3(client, bucket_name, image_data, image_name):
    client.put_object(Bucket=bucket_name, Key=image_name, Body=image_data)
