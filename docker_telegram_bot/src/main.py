import telebot
from only_model import ModelInference
TOKEN = 'YOUR_BOT_TOKEN'
MODEL_PATH = 'content/mushroom_logs/epoch_150.pth'

bot = telebot.TeleBot(TOKEN)
model_inference = ModelInference(model_path=MODEL_PATH)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне изображение, и я скажу тебе, что это.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Сохраняем файл локально
        with open('current_image.jpg', 'wb') as new_file:
            new_file.write(downloaded_file)

        # Выполняем инференс
        prediction = model_inference.inference('current_image.jpg')
        bot.reply_to(message, f'Я думаю, это {prediction}.')
    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {e}')

if __name__ == '__main__':
    bot.polling(none_stop=True)
