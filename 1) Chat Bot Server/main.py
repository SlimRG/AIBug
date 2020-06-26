# Библиотеки
import logging
import os
import smtplib
import sys
import telegram
import tempfile
import time

from configparser import ConfigParser
from PIL import Image
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton)
from telegram.ext import (Updater, CallbackQueryHandler, CommandHandler, ConversationHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler)
from datetime import datetime
from threading import Thread

# --------------------  НАСТРОЙКИ  ---------------------
# -- Telegram  
Token = '*******************'

# -- Programm
DataDir = 'data/' 
WelcomeSpeach = "Здравствуйте! \nЯ помогу вам ориентироваться в мире насекомых. \nВыберите, что вас интересует:"

# -- SMTP
HostSMTP = 'smtp.yandex.ru'
EmailSMTP = 'bbccaa@ya.ru'
LoginSMTP = 'bbccaa'
PasswordSMTP = '***********************'
ToEmailsSMTP = ['bbccaa@ya.ru', 'slimrg@ya.ru']

# -- Логирование
UseLogThread = True
LogFilePath = 'AIBugLogger.txt'
# ------------------------------------------------------

# Логирование
class LogThread(Thread):
    def __init__(self):
            Thread.__init__(self)
            self.name = 'LogT'
    
    def run(self):
        while True:
                with open(LogFilePath, 'r') as file:
                    read_file = file.read()
                    os.system('cls||clear')
                    print(read_file)
                    time.sleep(60)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, filename=LogFilePath)
logger = logging.getLogger(__name__)

if UseLogThread:
    LogThread = LogThread()
    LogThread.start()

# Авторизация
updater = Updater(token=Token, use_context=True) 
dispatcher = updater.dispatcher
logger.info('Bot service authorized('+Token+')')

# Обработка команд
def startCommand(update, context):
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(DataDir+'WelcomePage/WelcomeLogo.jpg', 'rb'), caption='')
    
    inline_keyboard = [
        [InlineKeyboardButton('Что это за насекомое?', callback_data='GoBug')], 
        [InlineKeyboardButton('Какое насекомое съело растение?', callback_data='GoPlant')], 
        [InlineKeyboardButton('Кто меня укусил?', callback_data='GoBite')], 
        [InlineKeyboardButton('Техподдержка', callback_data='GoSupport')]]

    markup = InlineKeyboardMarkup(inline_keyboard, resize_keyboard=True, one_time_keyboard=True)
    context.bot.send_message(chat_id=update.effective_chat.id, text=WelcomeSpeach, reply_markup=markup)
    logger.info("User %s session started", update.message.from_user.first_name)
    return "support_msg"

def process_support(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    msg = context.bot.send_message(chat_id, 'Пожалуйста опишите вашу проблему')
    return "support_text"

def process_bug(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id, 'Пришлите фотографию насекомого крупным планом, как на рисунке ниже:')
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(DataDir+'BugPage/ExImage.jpg', 'rb'), caption='')
    return "bug_page"

def process_plant(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id, 'Пришлите фотографию поврежденного растения крупным планом, как на рисунке ниже:')
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(DataDir+'PlantPage/ExImage.jpg', 'rb'), caption='')
    return "plant_page"

def process_bite(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id, 'Пришлите фотографию укуса крупным планом, как на рисунке ниже:')
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(DataDir+'BitePage/ExImage.jpg', 'rb'), caption='')
    return "bite_page"

def process_underconstruction(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id, 'Данная функция временно в разработке! \nПожалуйста, воспользуйтесь предложенным списком')
    time.sleep(5)
    return startCommand(update, context)

def send_email(body_text, emails):  
    host = HostSMTP
    from_addr = EmailSMTP

    
    BODY = "\r\n".join((
        "From: %s" % from_addr,
        "To: %s" % ', '.join(emails),
        "Subject: %s" % 'AIBug Support Request' ,
        "",
        body_text
    ))
    
    server = smtplib.SMTP_SSL(host)
    server.login(LoginSMTP, PasswordSMTP)
    server.sendmail(from_addr, emails, BODY.encode("ascii","ignore"))
    server.quit()

def askSupport(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    Report = "User: " + update.message.from_user.name + "\nDate: "+ str(update.message.date) + "\nText: \n" + update.message.text
    send_email(Report, ToEmailsSMTP)
    msg = context.bot.send_message(chat_id, "Спасибо! \nВ ближайшее время с вами свяжется администратор!")
    return startCommand(update, context)

def askBug(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    len_photo = len(update.message.photo)

    if (len_photo != 0):
        photo_file = context.bot.get_file(update.message.photo[-1].file_id)
        photo_name = update.message.photo[-1].file_path.split('/')[-1]
    else:
        photo_file = context.bot.get_file(update.message.document.file_id)
        photo_name = update.message.document.file_name

    photo_ext = photo_name.split('.')[-1]
    
    with tempfile.TemporaryDirectory() as temp:
        if (photo_ext == 'jpg'):
            photo_file.download(os.path.join(temp, 'bug.jpg'))
        else:
            photo_file.download(os.path.join(temp, photo_name))
            Image.open(os.path.join(temp, photo_name)).convert('RGB').save(os.path.join(temp, 'bug.jpg'),"JPEG")
        context.bot.send_message(chat_id, "Спасибо! \nФото получено. \nСобираю информацию...")
        # Обработка фото
        Result = 'Сервис обработки фото временно недоступен, попробуйте позже...'
        # -------
        context.bot.send_message(chat_id, Result)
    return startCommand(update, context)

def askPlant(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    len_photo = len(update.message.photo)

    if (len_photo != 0):
        photo_file = context.bot.get_file(update.message.photo[-1].file_id)
        photo_name = update.message.photo[-1].file_path.split('/')[-1]
    else:
        photo_file = context.bot.get_file(update.message.document.file_id)
        photo_name = update.message.document.file_name

    photo_ext = photo_name.split('.')[-1]
    
    with tempfile.TemporaryDirectory() as temp:
        if (photo_ext == 'jpg'):
            photo_file.download(os.path.join(temp, 'plant.jpg'))
        else:
            photo_file.download(os.path.join(temp, photo_name))
            Image.open(os.path.join(temp, photo_name)).convert('RGB').save(os.path.join(temp, 'plant.jpg'),"JPEG")
        context.bot.send_message(chat_id, "Спасибо! \nФото получено. \nСобираю информацию...")
        # Обработка фото
        Result = 'Сервис обработки фото временно недоступен, попробуйте позже...'
        # -------
        context.bot.send_message(chat_id, Result)
    return startCommand(update, context)

def askBite(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    len_photo = len(update.message.photo)

    if (len_photo != 0):
        photo_file = context.bot.get_file(update.message.photo[-1].file_id)
        photo_name = update.message.photo[-1].file_path.split('/')[-1]
    else:
        photo_file = context.bot.get_file(update.message.document.file_id)
        photo_name = update.message.document.file_name

    photo_ext = photo_name.split('.')[-1]
    
    with tempfile.TemporaryDirectory() as temp:
        if (photo_ext == 'jpg'):
            photo_file.download(os.path.join(temp, 'bite.jpg'))
        else:
            photo_file.download(os.path.join(temp, photo_name))
            Image.open(os.path.join(temp, photo_name)).convert('RGB').save(os.path.join(temp, 'bite.jpg'),"JPEG")
        context.bot.send_message(chat_id, "Спасибо! \nФото получено. \nСобираю информацию...")
        # Обработка фото
        Result = 'Сервис обработки фото временно недоступен, попробуйте позже...'
        # -------
        context.bot.send_message(chat_id, Result)
    return startCommand(update, context)

def wrongsupport(update, context):
    context.message.reply_text("Описание проблемы должно быть в текстовом виде! \nПопробуйте еще раз...")
    return startCommand(update, context)

def wrongimg(update, context):
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id, "Это не изображение! \nПопробуйте еще раз...")
    return startCommand(update, context)

# Хендлеры
start_command_handler = CommandHandler('start', startCommand)
support_callback_handler = CallbackQueryHandler(process_support, pattern='(GoSupport)')
bug_callback_handler = CallbackQueryHandler(process_bug, pattern='(GoBug)')
plant_callback_handler = CallbackQueryHandler(process_plant, pattern='(GoPlant)')
underconstruction_callback_handler = MessageHandler(Filters.all, process_underconstruction)
bite_callback_handler = CallbackQueryHandler(process_bite, pattern='(GoBite)')

support_message_handler = MessageHandler(Filters.text, askSupport)

bug_image_handler = MessageHandler(Filters.photo|Filters.document, askBug)
plant_image_handler = MessageHandler(Filters.photo|Filters.document, askPlant)
bite_image_handler = MessageHandler(Filters.photo|Filters.document, askBite)

wrongsupport_handler = MessageHandler(Filters.video | Filters.photo | Filters.document, wrongsupport)
wrongimg_handler = MessageHandler(Filters.all, wrongimg)

dialog_handler = ConversationHandler(entry_points=[start_command_handler],
                                     states={
                                          "home_page"   : [start_command_handler],

                                          "support_msg" : [support_callback_handler, bug_callback_handler, plant_callback_handler, bite_callback_handler, underconstruction_callback_handler],

                                          "support_text": [support_message_handler],

                                          "bug_page"    : [bug_image_handler, wrongimg_handler],

                                          "plant_page"  : [plant_image_handler, wrongimg_handler],

                                          "bite_page"   : [bite_image_handler, wrongimg_handler],
                                     },
                                     fallbacks=[wrongsupport_handler],
                                     per_message=False
                                    )


# Добавляем хендлеры в диспетчер
dispatcher.add_handler(dialog_handler)

# Начинаем поиск обновлений
updater.start_polling(clean=True)

# Останавливаем бота, если были нажаты Ctrl + C
updater.idle()


