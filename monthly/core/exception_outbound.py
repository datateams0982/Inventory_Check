import telegram
from pathlib import Path
from retry import retry
import os
import json

global config
config_path = Path(__file__).parent.parent / "config/basic_config.json"
if not os.path.exists(config_path):
    raise Exception(f'Configs not in this Directory: {config_path}')

with open(config_path, 'r') as fp:
    config = json.load(fp)

@retry(tries=config['telegram_retry']['tries'], delay=config['telegram_retry']['delay'])
def outbound(message, message_type='fail', **kwargs):

    bot = telegram.Bot(token=config['telegram_token'])
    chat_id_list = config['telegram_chatID']

    if message_type == 'fail':
        for chat_id in chat_id_list:
            bot.send_message(chat_id=chat_id, text=message)
    else:
        file_path = kwargs['file_path']
        for chat_id in chat_id_list:
            bot.send_message(chat_id=chat_id, text=message)
            bot.send_document(chat_id=chat_id, document=open(file_path, 'rb'))
