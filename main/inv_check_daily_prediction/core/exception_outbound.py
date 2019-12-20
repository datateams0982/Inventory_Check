import telegram
from pathlib import Path
from retry import retry
import os
import json
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
import subprocess

global config
config_path = Path(__file__).parent.parent / "basic_config.json"
if not os.path.exists(config_path):
    raise Exception(f'Configs not in this Directory: {config_path}')

with open(config_path, 'r') as fp:
    config = json.load(fp)

global button_name 
button_name = ['Rerun whole process', 'Reserve Model', 'Force feature engineering', 'Check DB correctness']

chat_id_list = config['telegram_chatID']


def start(bot, update):
    button_list = [[InlineKeyboardButton(item, callback_data=item)] for item in button_name]
    update.message.reply_text("What's Next",
        reply_markup = InlineKeyboardMarkup(button_list, n_col=1))

def start_process(bot, update):
    command = update.callback_query.data
    if command in ['Rerun whole process', 'Force feature engineering', 'Check DB correctness']:
        bot.editMessageText(message_id = update.callback_query.message.message_id,
                            chat_id = update.callback_query.message.chat.id,
                            text=f'Please specify the date wanted to {command}')
    else:
        bot.editMessageText(message_id = update.callback_query.message.message_id,
                            chat_id = update.callback_query.message.chat.id,
                            text=command)

def date_response(bot, update):
    bot.send_message(
        chat_id=update.effective_chat.id, text=update.message.text
        )


echo_handler = MessageHandler(Filters.text, date_response)
updater = Updater(token=config['telegram_token'])
updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(echo_handler)
updater.dispatcher.add_handler(CallbackQueryHandler(start_process))

updater.start_polling()
updater.idle()