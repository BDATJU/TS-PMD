import time
import json
import pika
import requests
import os
import sys
import argparse

# === 1. 路径与环境配置 ===
current_dir = os.path.dirname(os.path.abspath(__file__)) 
pmd_dir = os.path.dirname(current_dir)                  
if pmd_dir not in sys.path:
    sys.path.append(pmd_dir)
root_dir = os.path.dirname(pmd_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
project_path = os.path.abspath(os.path.join(os.getcwd(), "."))
if project_path not in sys.path:
    sys.path.append(project_path)

from design.inference import run_protein_design
from design.log import Logger

save_path = os.path.join(project_path, 'result')
if not os.path.exists(save_path): os.makedirs(save_path)
log = Logger(os.path.join(save_path, time.strftime("%Y%m%d") + '_design.log'))

MQ_USERNAME = "bda"      
MQ_PASSWORD = "dsp750403"      
MQ_HOST = "43.138.50.35"   
MQ_PORT = 5672 
POST_URL = "http://106.75.241.131:11985/BDAJAVA/task/result" 
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

def runDesignPrediction(params):
    config_info = params.get('configInfo', {})
    task_id = params.get('task_id')
    combined_params = {
        'task_id': task_id,
        **config_info,
        **params 
    }
    
    config = argparse.Namespace(**combined_params)
    config.static_dir = STATIC_DIR

    log.logger.info(f"Input={json.dumps(vars(config))}")
    try:
        ans = run_protein_design(config)
        res = {
            "task_id": f"{config.task_id}",
            "result": ans
        }
    except Exception as e: 
        log.logger.warning(f"There is something error! The message is {e}. Return None for Result")
        res = { 
            "task_id": f"{config.task_id}",
            "result": {"status": "error", "message": str(e)},
        }
    finally:
        log.logger.info(f"Result={res}")
    return res

def res2remote(res, post_result_url=POST_URL):
    try:
        log.logger.info(" [CONSUMER] The algorithm run successfully! Result={}".format(res))
        headers = {'Content-Type': 'application/json'}
        log.logger.info(" [CONSUMER] Send result to {}".format(post_result_url))
        response = requests.request("POST", post_result_url, headers=headers, data=json.dumps(res))
        log.logger.info(" [CONSUMER] Send Successfully! {}".format(response.text))
    except Exception as e:
        log.logger.warning("There is something error! The message is {}".format(e))

def callback_design(ch, method, properties, body):
    params = json.loads(body)
    log.logger.info(f"[CONSUMER] Received Message: {params}")

    try:
        res = runDesignPrediction(params)
        log.logger.info(f"[CONSUMER] The algorithm run successfully! Result={res}")
        res2remote(res)
        ch.basic_ack(delivery_tag=method.delivery_tag) 
    except Exception as e:
        log.logger.warning(f"There is something error! The message is {e}")

def consumer():
    task_name = "Protein Design (TS-PMD)"
    log.logger.info("==="*20)
    log.logger.info(f"[CONSUMER] This the worker for {task_name}!")
    queue_name = "queue_task_protein_design"
    callback_fn = callback_design
    
    credentials = pika.PlainCredentials(MQ_USERNAME, MQ_PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(MQ_HOST, MQ_PORT, 
                                  credentials=credentials,
                                  heartbeat=30*60))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name, durable=True)

    channel.basic_qos(prefetch_count=1)  
    channel.basic_consume(queue=queue_name,
                          auto_ack=False, 
                          on_message_callback=callback_fn)

    log.logger.info('[CONSUMER] Waiting for message. To exit press CTRL+C')
    channel.start_consuming()
    log.logger.info("==="*20 + '\n')

if __name__ == '__main__':
    consumer()
