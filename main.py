from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

from config import TOKEN
from handlers import button_handler, button_query_handler, choice_handler, handle_message, start, stop

def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^(compose_request|leave_feedback|refine_query|generate_new_query)$'))
    application.add_handler(CallbackQueryHandler(button_query_handler,pattern='^(next_page|prev_page)$'))
    application.add_handler(CallbackQueryHandler(choice_handler, pattern=r'^choose_\d+$'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("started")
    application.run_polling()


if __name__ == '__main__':
    main()
