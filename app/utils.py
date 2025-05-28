import re

import asyncio
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from keyboards import create_choice_keyboard#, create_query_buttons
from logger import logger
from response import generate_gpt_response, search_response, fake_generate_gpt_response, \
    fake_search_response


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
    user = update.effective_user
    logger.info(f"User {user.id} started LLM generate (3)")

    query = update.callback_query
    user_id = update.effective_user.id

    if "locks" not in context.bot_data:
        context.bot_data["locks"] = {}

    if user_id not in context.bot_data["locks"]:
        context.bot_data["locks"][user_id] = asyncio.Lock()

    try:
        async with context.bot_data["locks"][user_id]:
            context.user_data["active_query"] = True

            #gpt_response = await fake_generate_gpt_response(document_id, context, url)
            gpt_response = await generate_gpt_response(document_id, context, url)
            logger.info(f"bot got users {user.id} LLM generate (3-middle)")

            answer = gpt_response.get('roadmap', "Ошибка: ответ не получен.")
            answer = convert_markdown_to_html(answer)

            context.user_data["current_select"] = formatted_selected_service

            await query.edit_message_text(
                text=f"<b>Ответ:</b>\n\n{answer}",
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        logger.info(f"User {user.id} reached Exception in fetch_gpt_and_edit: {e} ;(3e)")
        await query.edit_message_text(
            "Произошла ошибка при получении ответа. Попробуйте снова.",
            parse_mode=ParseMode.HTML
        )

    finally:
        context.user_data["active_query"] = False
        logger.info(f"User {user.id} got LLM generate (3-end) in messenger")


async def search_and_edit(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    user = update.effective_user
    try:

        #response = await fake_search_response(context, url)
        response = await search_response(context, url)
        results = response

        # if len(results) == 0:
        #     await handle_empty_results(update, context)
        #     return

        context.user_data.update({
            'query_attempts': 0,
            'results': results,
            'current_page': 0,
            'indices': [result[0] for result in get_page_results(results, 0)]
        })

        page_results = get_page_results(results, 0)
        flag = len(results) > 5
        choice_keyboard = create_choice_keyboard(context.user_data['indices'], flag)

        response_text = "\n\n".join(
            f"{i + 1}. {result[1].split('`')[0]}"
            for i, result in enumerate(page_results)
        )

        logger.info(f"User {user.id} ended search (1-end)")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Найденные похожие записи:\n\n{response_text}",
            reply_markup=choice_keyboard
        )

    except Exception as e:
        logger.info(f"User {user.id} reached Exception in search_and_edit: {e} ;(1e)")
        print(f"Ошибка в search_and_edit: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Произошла ошибка при поиске услуг. Пожалуйста, попробуйте позже."
        )

async def handle_empty_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query_attempts = context.user_data.get('query_attempts')
    refine_mode = context.user_data.get('refine_mode')
    current_query = context.user_data.get('user_query', '')

    if refine_mode:

        new_query_part = update.message.text
        updated_query = f"{current_query}{new_query_part}".strip()
        context.user_data['user_query'] = updated_query
        context.user_data['query_attempts'] = query_attempts + 1
        await update.message.reply_text(
            f"Ваш текущий уточненный запрос: {updated_query}"
        )

    else:
        current_query = update.message.text
        context.user_data['user_query']=current_query


    # if query_attempts < 2:
    #     if not refine_mode:
    #         await update.message.reply_text(
    #             f"Мы не нашли похожих тем. Вы можете уточнить свой запрос или попробовать создать новый, ваш текущий запрос: {current_query}",
    #             reply_markup=create_query_buttons()
    #         )
    if query_attempts < 2:
        if not refine_mode:
            await update.message.reply_text(
                f"Мы не нашли похожих тем. Задайте запрос заново.",
                reply_markup=None
            )
    else:
        context.user_data['query_attempts'] = 0
        context.user_data['user_query'] = ""
        context.user_data['refine_mode'] = False
        await update.message.reply_text("Ваш запрос не дал результатов. Пожалуйста, перепишите запрос.")