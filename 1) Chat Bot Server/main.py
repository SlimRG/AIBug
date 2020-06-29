# Библиотеки
import logging
import os
import re
import smtplib
import sys
import telegram
import tempfile
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from efficientnet_pytorch import EfficientNet
from configparser import ConfigParser
from PIL import Image
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton)
from telegram.ext import (Updater, CallbackQueryHandler, CommandHandler, ConversationHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler)
from datetime import date, datetime
from threading import Thread
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

class enet(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enet, self).__init__()
        self.enet = EfficientNet.from_pretrained(backbone)
        
        for param in self.enet.parameters():
            param.requires_grad = False
    
        #for param in self.myfc.parameters():
        #    param.requires_grad = True
        self.l1 = nn.Linear(124416 ,1244 )
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(1244,out_dim) # 6 is number of classes
        self.relu = nn.LeakyReLU()
        
    def forward(self, input):
        x = self.enet.extract_features(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return nn.functional.softmax(x)


# --------------------  НАСТРОЙКИ  ---------------------
# -- Telegram  
Token = '*****'

# -- Programm
DataDir = 'data/' 
WelcomeSpeach = "Здравствуйте! \nЯ помогу вам ориентироваться в мире насекомых. \nВыберите, что вас интересует:"

# -- SMTP
HostSMTP = 'smtp.yandex.ru'
EmailSMTP = 'bbccaa@ya.ru'
LoginSMTP = 'bbccaa'
PasswordSMTP = '*****'
ToEmailsSMTP = ['bbccaa@ya.ru', 'slimrg@ya.ru']

# -- Logging
UseLogThread = False
LogFilePath = 'logs'
RemoveLogsAfter = 7; # Days

# -- DataBase 
DBFolder = 'dictionaries'
# ------------------------------------------------------

# Автоочистка
def del_old_files(path, min_days=3, recursive=False):
    p = Path(path)
    glob_pattern = "**/*" if recursive else "*"
    for f in p.glob(glob_pattern):
        if (f.is_file()
            and
            (datetime.now() - datetime.fromtimestamp(f.stat().st_ctime)).days >= min_days):
            f.unlink()
del_old_files(LogFilePath, 3, True)

# Логирование
if not os.path.exists(LogFilePath):
    os.makedirs(LogFilePath)
LogPath = os.path.join(LogFilePath, str(date.today())+'-'+str(datetime.now().hour)+'-'+str(datetime.now().minute)+'-'+str(datetime.now().second)+'_log.txt')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.INFO, 
                    filename=LogPath)
logger = logging.getLogger(__name__)

# Логирование в консоли
class LogThread(Thread):
    def __init__(self):
            Thread.__init__(self)
            self.name = 'LogT'
    
    def run(self):
        while True:
                with open(LogPath, 'r') as file:
                    read_file = file.read()
                    os.system('cls||clear')
                    print(read_file)
                    time.sleep(60)
if UseLogThread:
    LogThread = LogThread()
    LogThread.start()

# Получение списка баз данных (насекомые)
DBBugs = []
DBBugsPath = os.path.join(DBFolder, 'bugs')
for something in os.listdir(DBBugsPath):
    if os.path.isdir(os.path.join(DBBugsPath, something)):
        DBBugs.append(something)

DBBites = []
DBBitesPath = os.path.join(DBFolder, 'bites')
for something in os.listdir(DBBitesPath):
    if os.path.isdir(os.path.join(DBBitesPath, something)):
        DBBites.append(something)

logger.info('Bugs:  '+ str(DBBugs))      
logger.info('Bites: '+ str(DBBites))

# Авторизация
updater = Updater(token=Token, use_context=True) 
dispatcher = updater.dispatcher
logger.info('Bot service authorized('+Token+')')

# Парсинг БД
def send_info(update, context, maindir, mainfile):
    query = update.callback_query
    chat_id = update.effective_chat.id

    # Разбиение текста
    with open(os.path.join(maindir, mainfile)) as f:
        dirtytext = re.split(r'\[(.*?)\]', f.read())

    posttext = ""
    for text in dirtytext:
        # Если картинка
        if os.path.exists(os.path.join(maindir, text)):
            if posttext: 
                context.bot.send_message(chat_id, posttext, parse_mode= "Markdown")
                posttext = ""
            context.bot.send_photo(chat_id, photo=open(os.path.join(maindir, text), 'rb'), caption='')
        # Если гиперссылка
        elif (text.find('http://') != -1) or (text.find('https://') != -1):
            text = posttext + ']' + text[0:-1]
            context.bot.send_message(chat_id, text, parse_mode= "Markdown")    
            posttext = ""   
        # Если простой текст
        else: 
            posttext += text

    if posttext: 
                context.bot.send_message(chat_id, posttext, parse_mode= "Markdown")

# Обработка команд
def send_buginfo(update, context, bugname):
    query = update.callback_query
    chat_id = update.effective_chat.id
    if bugname not in DBBugs:
        context.bot.send_message(chat_id, "Информация о насекомом не найдена! \nВ будущем база данных будет расширяться...")
    else:
        send_info(update, context, os.path.join(DBBugsPath, bugname), bugname+'.txt')

def send_biteinfo(update, context, bitename):
    query = update.callback_query
    chat_id = update.effective_chat.id
    if bitename not in DBBites:
        context.bot.send_message(chat_id, "Информация о укусе не найдена! \nВ будущем база данных будет расширяться...")
    else:
        send_info(update, context, os.path.join(DBBitesPath, bitename), bitename+'.txt')
        
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
    context.bot.send_photo(chat_id, photo=open(DataDir+'BugPage/ExImage.jpg', 'rb'), caption='')
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
    logger.warn('Using underconstruction functions by "%s", in context "%s"' % (update.message.from_user.name, context))
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
        photo_name = photo_file.file_path.split('/')[-1]
    else:
        photo_file = context.bot.get_file(update.message.document.file_id)
        photo_name = update.message.document.file_name

    photo_ext = photo_name.split('.')[-1]
    
    with tempfile.TemporaryDirectory() as temp:
        if (photo_ext == 'jpg'):
            photo_file.download(os.path.join(temp, 'bug.jpg'))
        elif (photo_ext in ['bmp', 'tiff', 'tif', 'gif', 'jpeg', 'png']):
            photo_file.download(os.path.join(temp, photo_name))
            Image.open(os.path.join(temp, photo_name)).convert('RGB').save(os.path.join(temp, 'bug.jpg'),"JPEG")
        else:
            logger.error('Unknown photo format "%s" by user "%s"' % (photo_ext, update.message.from_user.name))
            return wrongimg(update, context)
        context.bot.send_message(chat_id, "Спасибо! \nФото получено. \nСобираю информацию...")
        # Обработка фото
        model_name = 'efficientnet-b3'
        n_class = 14
        model = enet(model_name, n_class)
        model.load_state_dict(torch.load('neuron/bugs/model_bugs_statedict', map_location=torch.device('cpu')))
        model.eval()

        # Preprocess image
        image_size = 300
        # изменения
        tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        # загрузка как тензор
        img = tfms(Image.open(os.path.join(temp, 'bug.jpg'))).unsqueeze(0)
        # Load ImageNet class names
        labels_map = json.load(open('neuron/bugs/labels_map.json'))
        new_labels_map = {}
        for key in labels_map.keys():
            new_labels_map[labels_map[key]] = key
        
        model.eval()
        with torch.no_grad():
            img = img.to('cpu')
            outputs = model(img)[0]

        Results = []
        StrB = 'Это: \n'
        for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
                StrB += str(new_labels_map[idx]) + ": " + str(outputs[idx].item()*100) + "\n"
                Results.append(new_labels_map[idx])

        context.bot.send_message(chat_id, StrB)
        send_buginfo(update, context, Results[0])
    return startCommand(update, context)

def askPlant(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    len_photo = len(update.message.photo)

    if (len_photo != 0):
        photo_file = context.bot.get_file(update.message.photo[-1].file_id)
        photo_name = photo_file.file_path.split('/')[-1]
    else:
        photo_file = context.bot.get_file(update.message.document.file_id)
        photo_name = update.message.document.file_name

    photo_ext = photo_name.split('.')[-1]
    
    with tempfile.TemporaryDirectory() as temp:
        if (photo_ext == 'jpg'):
            photo_file.download(os.path.join(temp, 'plant.jpg'))
        elif (photo_ext in ['bmp', 'tiff', 'tif', 'gif', 'jpeg', 'png']):
            photo_file.download(os.path.join(temp, photo_name))
            Image.open(os.path.join(temp, photo_name)).convert('RGB').save(os.path.join(temp, 'plant.jpg'),"JPEG")
        else:
            logger.error('Unknown photo format "%s" by user "%s"' % (photo_ext, update.message.from_user.name))
            return wrongimg(update, context)
        context.bot.send_message(chat_id, "Спасибо! \nФото получено. \nСобираю информацию...")
        # Обработка фото
        model_name = 'efficientnet-b3'
        n_class = 5
        model = enet(model_name, n_class)
        model.load_state_dict(torch.load('neuron/plants/model_plants_statedict', map_location=torch.device('cpu')))
        model.eval()

        # Preprocess image
        image_size = 300
        # изменения
        tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        # загрузка как тензор
        img = tfms(Image.open(os.path.join(temp, 'plant.jpg'))).unsqueeze(0)
        # Load ImageNet class names
        labels_map = json.load(open('neuron/plants/labels_map_plants.json'))
        new_labels_map = {}
        for key in labels_map.keys():
            new_labels_map[labels_map[key]] = key
        
        model.eval()
        with torch.no_grad():
            img = img.to('cpu')
            outputs = model(img)[0]

        Results = []
        StrB = 'Это: \n'
        for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
                StrB += str(new_labels_map[idx]) + ": " + str(outputs[idx].item()*100) + "\n"
                Results.append(new_labels_map[idx])

        context.bot.send_message(chat_id, StrB)
        send_buginfo(update, context, Results[0])
    return startCommand(update, context)

def askBite(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    len_photo = len(update.message.photo)

    if (len_photo != 0):
        photo_file = context.bot.get_file(update.message.photo[-1].file_id)
        photo_name = photo_file.file_path.split('/')[-1]
    else:
        photo_file = context.bot.get_file(update.message.document.file_id)
        photo_name = update.message.document.file_name

    photo_ext = photo_name.split('.')[-1]
    
    with tempfile.TemporaryDirectory() as temp:
        if (photo_ext == 'jpg'):
            photo_file.download(os.path.join(temp, 'bite.jpg'))
        elif (photo_ext in ['bmp', 'tiff', 'tif', 'gif', 'jpeg', 'png']):
            photo_file.download(os.path.join(temp, photo_name))
            Image.open(os.path.join(temp, photo_name)).convert('RGB').save(os.path.join(temp, 'bite.jpg'),"JPEG")
        else:
            logger.error('Unknown photo format "%s" by user "%s"' % (photo_ext, update.message.from_user.name))
            return wrongimg(update, context)
        context.bot.send_message(chat_id, "Спасибо! \nФото получено. \nСобираю информацию...")
        # Обработка фото
        model_name = 'efficientnet-b3'
        n_class = 4
        model = enet(model_name, n_class)
        model.load_state_dict(torch.load('neuron/bites/model_bites_statedict', map_location=torch.device('cpu')))
        model.eval()

        # Preprocess image
        image_size = 300
        # изменения
        tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        # загрузка как тензор
        img = tfms(Image.open(os.path.join(temp, 'bite.jpg'))).unsqueeze(0)
        # Load ImageNet class names
        labels_map = json.load(open('neuron/bites/labels_map.json'))
        new_labels_map = {}
        for key in labels_map.keys():
            new_labels_map[labels_map[key]] = key
        
        model.eval()
        with torch.no_grad():
            img = img.to('cpu')
            outputs = model(img)[0]

        Results = []
        StrB = 'Это: \n'
        for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
                StrB += str(new_labels_map[idx]) + ": " + str(outputs[idx].item()*100) + "\n"
                Results.append(new_labels_map[idx])

        context.bot.send_message(chat_id, StrB)
        send_biteinfo(update, context, Results[0])
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
