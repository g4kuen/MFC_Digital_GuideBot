from os import remove

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, \
    CallbackContext

from telegram.constants import ParseMode

from config import url
from utils import get_page_results, convert_markdown_to_html
from keyboards import generate_choice_keyboard, create_query_buttons
from response import search_response, generate_gpt_response
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('waiting_for_feedback'):
        feedback = update.message.text
        ###обработка для feedback
        await update.message.reply_text(f"Вы составили отзыв: {feedback}")
        context.user_data['waiting_for_feedback'] = False

    if context.user_data.get('waiting_for_query'):
        user_message = update.message.text
        if 'query_attempts' not in context.user_data:
            context.user_data['query_attempts'] = 1
        if 'user_query' in context.user_data:
            user_message = context.user_data['user_query'] + " " + user_message
        if 'user_query' not in context.user_data or context.user_data['user_query'] == "":
            context.user_data['user_query']=user_message

        flag = False

        response = await search_response(context, url)
        results = response


        if len(results) == 0:
            await handle_empty_results(update, context)
        else:
            context.user_data['user_query'] = ""
            context.user_data['query_attempts'] = 0

            context.user_data['results'] = results
            context.user_data['current_page'] = 0

            page_results = get_page_results(results, context.user_data['current_page'])
            page_indices = [result[0] for result in page_results]  # Индексы для текущей страницы
            context.user_data['indices'] = page_indices


            if (len(results) > 5):
                flag = True
            else:
                flag = False

            choice_keyboard = generate_choice_keyboard(page_indices, flag)
         #   context.user_data['refine_mode'] = False

            response = "\n\n".join([f"{i + 1}. {result[1].split('`')[0]}" for i, result in enumerate(page_results)])

            await update.message.reply_text(
                text=f"Найденные похожие записи:\n\n{response}",
                reply_markup=choice_keyboard if choice_keyboard else None,

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


    if query_attempts < 2:
        if not refine_mode:
            await update.message.reply_text(
                f"Мы не нашли похожих тем. Вы можете уточнить свой запрос или попробовать создать новый, ваш текущий запрос: {current_query}",
                reply_markup=create_query_buttons()
            )
    else:
        context.user_data['query_attempts'] = 0
        context.user_data['user_query'] = ""
        context.user_data['refine_mode'] = False
        await update.message.reply_text("Ваш запрос не дал результатов. Пожалуйста, перепишите запрос.")

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

        formatted_selected_service = convert_markdown_to_html(selected_service[1])

        async def fetch_gpt_and_edit():
            try:
                gpt_response = await generate_gpt_response(document_id, context, url)
                answer = gpt_response.get('roadmap', "Ошибка: ответ не получен.")
                context.user_data["current_select"] = formatted_selected_service
                formatted_answer = convert_markdown_to_html(answer)

                await query.edit_message_text(
                    text=f"<b>Вы выбрали услугу</b>: {formatted_selected_service} \n\n{formatted_answer}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=None
                )
                context.user_data["is_gpt_active"] = False

            except Exception as e:
                print(f"Ошибка в fetch_gpt_and_edit: {e}")
                await query.edit_message_text("Произошла ошибка при получении ответа. Попробуйте снова.")

        if context.user_data["is_gpt_active"]:
            await query.edit_message_text(
                text=f"<b>Вы уже выбрали услугу</b>: {context.user_data["current_select"]} \n\nПожалуйста, подождите прошлого ответа",
                parse_mode=ParseMode.HTML,
                reply_markup=None
            )
        else:
            await query.edit_message_text(
                text=f"<b>Вы выбрали услугу</b>: {formatted_selected_service} \n\nПожалуйста, подождите ответа",
                parse_mode=ParseMode.HTML,
                reply_markup=None
            )
            context.user_data["is_gpt_active"] = True
            asyncio.create_task(fetch_gpt_and_edit())
#

    except Exception as e:
        print(f"Ошибка в choice_handler: {e}")
        await query.edit_message_text("Произошла ошибка. Попробуйте снова.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Составить запрос ", callback_data='compose_request')],
        [InlineKeyboardButton("Оставить отзыв ", callback_data='leave_feedback')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['waiting_for_query'] = False
    await update.message.reply_text("Вы остановили прием запросов, для запуска выберите 'Составить запрос' в команде /start")
