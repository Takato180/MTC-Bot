from kafka import KafkaProducer
import json, gzip, os
""""
ファイル: producer.py
概要:
    このモジュールは、KafkaProducerをセットアップし、データの取り込みと指定されたKafkaトピックへのメッセージ送信を行います。
    プロデューサーは以下の設定で構成されています:
        - bootstrap_servers: 環境変数 "KAFKA_BOOTSTRAP" から取得し、デフォルトは "localhost:19092" です。
        - value_serializer: メッセージをJSONに変換し、エンコード後にgzip圧縮するラムダ関数を使用します。
        - linger_ms: 5ミリ秒に設定し、メッセージのバッチ処理を可能にします。
"""

producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:19092"),
    value_serializer=lambda v: gzip.compress(json.dumps(v).encode()),
    linger_ms=5,
)

"""
関数:
    send(topic: str, msg: dict)
        指定されたKafkaトピックにメッセージを送信します。
        パラメータ:
            topic (str): メッセージ送信先のKafkaトピック。
            msg (dict): 辞書形式で提供される送信データ。
        挙動:
            - 送信するトピックとメッセージを表示します。
            - 設定されたKafkaProducerを使用してメッセージを送信します。
            - メッセージ送信時に発生したKafkaのエラーを表示するエラーバックを登録します。
使用例:
    send("example_topic", {"example_key": "example_value"})
"""
def send(topic: str, msg: dict):
    print("SEND", topic, msg)
    # producer.send(topic, msg).add_errback(lambda e: print("Kafka error:", e))
    producer.send(topic, msg).add_errback(lambda e: print("Kafka error:", e))
