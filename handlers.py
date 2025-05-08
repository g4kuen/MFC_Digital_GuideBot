from os import remove

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, \
    CallbackContext

from telegram.constants import ParseMode

from MFC_Digital_GuideBot.logger import logger
from config import url
from utils import fetch_gpt_and_edit, search_and_edit, get_page_results
from keyboards import generate_choice_keyboard, create_query_buttons
from response import search_response, fake_search_response
import asyncio

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()


    if query.data == 'compose_request':
        await query.edit_message_text(text="Составьте запрос. Чтобы остановить прием запросов, выполните команду /stop")
        context.user_data['waiting_for_query'] = True
    if query.data == 'leave_feedback':
        await query.edit_message_text(text="Составьте отзыв.",)
        context.user_data['waiting_for_query'] = False
        context.user_data['waiting_for_feedback'] = True
    if query.data == 'refine_query':
        context.user_data['refine_mode'] = True
        await query.edit_message_text(text=f"Пожалуйста, уточните ваш запрос, добавив дополнительную информацию. текущий запрос : {context.user_data['user_query']} ")
    elif query.data == 'generate_new_query':
        context.user_data['refine_mode'] = False
        context.user_data['query_attempts'] = 0
        context.user_data['user_query'] = ""
        await query.edit_message_text(text="Пожалуйста, введите новый запрос.")


# async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     print("1")
#     if context.user_data.get('waiting_for_feedback'):
#         feedback = update.message.text
#         ###обработка для feedback
#         await update.message.reply_text(f"Вы составили отзыв: {feedback}")
#         context.user_data['waiting_for_feedback'] = False
#
#     if context.user_data.get('waiting_for_query'):
#         user_message = update.message.text
#         context.user_data['user_query'] = user_message
#         if 'query_attempts' not in context.user_data:
#             context.user_data['query_attempts'] = 1
#         if 'user_query' in context.user_data:
#             user_message = context.user_data['user_query'] + " " + user_message
#         if 'user_query' not in context.user_data or context.user_data['user_query'] == "":
#             context.user_data['user_query'] = user_message
#
#         flag = False
#
#         response = await search_response(context, url)
#         #response = await fake_search_response(context, url)
#         results = response
#
#
#         if len(results) == 0:
#             await handle_empty_results(update, context)
#         else:
#
#             context.user_data['query_attempts'] = 0
#
#             context.user_data['results'] = results
#             context.user_data['current_page'] = 0
#
#             page_results = get_page_results(results, context.user_data['current_page'])
#             page_indices = [result[0] for result in page_results]  # Индексы для текущей страницы
#             context.user_data['indices'] = page_indices
#
#
#             if (len(results) > 5):
#                 flag = True
#             else:
#                 flag = False
#
#             choice_keyboard = generate_choice_keyboard(page_indices, flag)
#          #   context.user_data['refine_mode'] = False
#
#             response = "\n\n".join([f"{i + 1}. {result[1].split('`')[0]}" for i, result in enumerate(page_results)])
#
#             await update.message.reply_text(
#                 text=f"Найденные похожие записи:\n\n{response}",
#                 reply_markup=choice_keyboard if choice_keyboard else None,
#
#             )
#             print("1-end")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user


    if context.user_data.get('waiting_for_feedback'):
        feedback = update.message.text
        await update.message.reply_text(f"Вы составили отзыв: {feedback}")
        context.user_data['waiting_for_feedback'] = False
        return

    if context.user_data.get('waiting_for_query'):
        user_message = update.message.text
        context.user_data['user_query'] = user_message

        if 'query_attempts' not in context.user_data:
            context.user_data['query_attempts'] = 1

        logger.info(f"User {user.id} started search (1)")
        await update.message.reply_text("🔍 Идет поиск подходящих услуг...")

        asyncio.create_task(
            search_and_edit(update, context, url)
        )
        logger.info(f"User {user.id} task started in background (1-middle)")


async def button_query_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    max_page = len(context.user_data['results']) // 5

    if query.data == "prev_page":
        if context.user_data['current_page'] > 0:
            context.user_data['current_page'] -= 1
        else:
            context.user_data['current_page'] = max_page

    elif query.data == "next_page":
        if context.user_data['current_page'] < max_page:
            context.user_data['current_page'] += 1
        else:
            context.user_data['current_page'] = 0

    current_page = context.user_data['current_page']
    results = context.user_data['results']
    page_results = results[current_page * 5:(current_page + 1) * 5]
    page_indices = [result[0] for result in page_results]
    context.user_data['indices'] = page_indices

    response = "\n\n".join([f"{i + 1}. {result[1].split('`')[0]}" for i, result in enumerate(page_results)])

    choice_keyboard=generate_choice_keyboard(page_indices, True)


    await query.edit_message_text(
        text=f"Найденные похожие записи:\n\n{response}",
        reply_markup=choice_keyboard if choice_keyboard else None
    )


async def choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"User {user.id} started choice (2)")

    query = update.callback_query
    try:
        await query.answer()

        indices = context.user_data.get('indices', [])
        if not indices:
            await query.edit_message_text("Ошибка: нет доступных индексов для выбора.")
            return

        choice_number = int(query.data.split('_')[1]) - 1
        if choice_number < 0 or choice_number >= len(indices):
            await query.edit_message_text("Ошибка: неверный выбор.")
            return

        selected_service = context.user_data['search_results'][indices[choice_number]]
        document_id = context.user_data['search_id'][indices[choice_number]]


        logger.info(f"User {user.id} ended choice (2-end)")
        if "active_query" not in context.user_data or not context.user_data["active_query"]:
            context.user_data["current_select"] = selected_service[1]

            await query.edit_message_text(
                text=f"<b>Вы выбрали услугу</b>: {selected_service[1]} \n\nПожалуйста, подождите ответа",
                parse_mode=ParseMode.HTML,
                reply_markup=None
            )
            asyncio.create_task(fetch_gpt_and_edit(update, context, selected_service[1], document_id, url))
        else:

            await query.edit_message_text(
                text=f"<b>Вы уже выбрали услугу</b>: {context.user_data['current_select']} \n\nПожалуйста, подождите прошлого ответа",
                parse_mode=ParseMode.HTML,
                reply_markup=None
            )
            logger.info(f"User {user.id} ended choice, his generate not ended (2-end), generate prevented")


    except Exception as e:
        logger.info(f"User {user.id} reached Exception in choice_handler: {e} ;(2e)")
        await query.edit_message_text("Произошла ошибка при обработке выбора.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"User {user.id} started the bot")
    keyboard = [
        [InlineKeyboardButton("Составить запрос ", callback_data='compose_request')],
        [InlineKeyboardButton("Оставить отзыв ", callback_data='leave_feedback')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['waiting_for_query'] = False
    await update.message.reply_text("Вы остановили прием запросов, для запуска выберите 'Составить запрос' в команде /start")
