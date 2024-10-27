import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# Загрузка переменных окружения из .env файла
load_dotenv()
TOKEN = os.getenv("TOKEN")

# Переменная для хранения последнего сообщения пользователя
last_message = ""

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Составить запрос", callback_data='compose_request')],
        [InlineKeyboardButton("Оставить отзыв", callback_data='leave_feedback')]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)

# Обработчик нажатий на кнопки
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == 'compose_request':
        await query.edit_message_text(text="Вы составили запрос!")
    elif query.data == 'leave_feedback':
        await query.edit_message_text(text="Составьте отзыв.")
        # Устанавливаем состояние ожидания отзыва
        context.user_data['waiting_for_feedback'] = True

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Проверяем, ожидает ли бот отзыв
    if context.user_data.get('waiting_for_feedback'):
        feedback = update.message.text
        await update.message.reply_text(f"Вы составили отзыв: {feedback}")
        # Сбрасываем состояние ожидания
        context.user_data['waiting_for_feedback'] = False
    else:
        await update.message.reply_text("Пожалуйста, используйте кнопки для выбора действия.")

# Основная функция для запуска бота
def main():
    # Создание приложения и регистрация обработчиков
    application = Application.builder().token(TOKEN).build()

    # Обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    application.add_handler(CallbackQueryHandler(button_handler))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == '__main__':
    main()
