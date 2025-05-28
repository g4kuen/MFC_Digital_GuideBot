from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from logger import logger

from config import TOKEN
from handlers import button_handler, choice_handler, message_handler, start, stop, format_handler#, button_query_handler

def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    #application.add_handler(CallbackQueryHandler(button_handler, pattern='^(compose_request|leave_feedback|refine_query|generate_new_query)$'))
    application.add_handler(CallbackQueryHandler(button_handler,pattern='^(compose_request|leave_feedback)$'))

    #application.add_handler(CallbackQueryHandler(button_query_handler,pattern='^(next_page|prev_page)$'))
    application.add_handler(CallbackQueryHandler(format_handler, pattern=r'^format_(long|short)$'))
    application.add_handler(CallbackQueryHandler(choice_handler, pattern=r'^choose_\d+$'))




    logger.info("BOT STARTED")
    application.run_polling()


if __name__ == '__main__':
    main()
