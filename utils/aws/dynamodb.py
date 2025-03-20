def put_item_to_dynamodb(item, table):
    """
    DynamoDBにアイテムを追加する関数
    :param item: 辞書型のアイテム
    """
    try:
        response = table.put_item(Item=item)
        print("PutItem succeeded:", response)
    except Exception as e:
        print("Error putting item:", e)
