import re

import asyncio
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes


from MFC_Digital_GuideBot.response import generate_gpt_response, fake_generate_gpt_response


def split_message(message, max_length=4096):
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]

def split_and_escape_message(message, max_length=4096):
    escaped_message = convert_markdown_to_html(message)
    return split_message(escaped_message, max_length)


def get_page_results(results, page, page_size=5):
    start = page * page_size
    end = start + page_size
    return results[start:end]

def convert_markdown_to_html(markdown_text):
    bold_pattern = re.compile(r'\*\*(.*?)\*\*')

    def replace_bold(match):
        return f'<b>{match.group(1)}</b>'

    html_text = bold_pattern.sub(replace_bold, markdown_text)

    return html_text


async def fetch_gpt_and_edit(update: Update, context: ContextTypes.DEFAULT_TYPE,
                             formatted_selected_service: str, document_id: str, url: str) -> None:
    print("3 - Начало обработки GPT запроса")
    query = update.callback_query
    user_id = update.effective_user.id

    # Инициализация системы блокировок
    if "locks" not in context.bot_data:
        context.bot_data["locks"] = {}

    if user_id not in context.bot_data["locks"]:
        context.bot_data["locks"][user_id] = asyncio.Lock()

    try:
        # Блокировка для этого пользователя
        async with context.bot_data["locks"][user_id]:
            context.user_data["active_query"] = True

            # Имитация долгого запроса к GPT
            gpt_response = await fake_generate_gpt_response(document_id, context, url)
            print("2 - Получен ответ от GPT")

            answer = gpt_response.get('roadmap', "Ошибка: ответ не получен.")
            answer = convert_markdown_to_html(answer)

            # Сохраняем текущий выбор пользователя
            context.user_data["current_select"] = formatted_selected_service

            await query.edit_message_text(
                text=f"<b>Ответ:</b>\n\n{answer}",
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        print(f"Ошибка в fetch_gpt_and_edit: {e}")
        await query.edit_message_text(
            "Произошла ошибка при получении ответа. Попробуйте снова.",
            parse_mode=ParseMode.HTML
        )

    finally:
        # Всегда снимаем флаг активности
        context.user_data["active_query"] = False
        print("3-end - Завершение обработки GPT запроса")