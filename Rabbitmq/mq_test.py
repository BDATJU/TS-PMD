import pika
import json

rabbitmq_config = {
    'host': '43.138.50.35',
    'port': 5672,
    'username': 'bda',
    'password': 'dsp750403',
    'virtual_host': '/'
}

credentials = pika.PlainCredentials(rabbitmq_config['username'], rabbitmq_config['password'])
parameters = pika.ConnectionParameters(
    host=rabbitmq_config['host'],
    port=rabbitmq_config['port'],
    virtual_host=rabbitmq_config['virtual_host'],
    credentials=credentials
)

try:
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    task_payload = {
        "task_id": "252", 
        "pdb_id": "1CLW", 
        "chain_id": "A",
        #  "num_steps": 500, 可选：在线上可调低 Step 数加快出图速度
        "threshold": 0.5
    }
    
    channel.basic_publish(
        exchange="exchange_task",
        routing_key="routing_key_protein_design",  
        body=json.dumps(task_payload),
        properties=pika.BasicProperties(delivery_mode=2)
    )

    print(f"✅ 成功发送蛋白质智能设计任务:\n{json.dumps(task_payload, indent=2)}")
    connection.close()
except Exception as e:
    print(f"发送失败: {e}")