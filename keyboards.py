from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup


def generate_choice_keyboard(indices,flag):

    keyboard=[]
    if flag == True:
        keyboard = [
            [InlineKeyboardButton("⬅️ Назад", callback_data="prev_page"),
             InlineKeyboardButton("Вперёд ➡️", callback_data="next_page")]
        ]

        keyboard += [
            [InlineKeyboardButton(f"Ответ {i + 1}", callback_data=f"choose_{i + 1}")]
            for i in range(len(indices))
        ]
    else:
        keyboard = [[ InlineKeyboardButton(f"Ответ {i + 1}", callback_data=f"choose_{i + 1}")]
            for i in range(len(indices))
        ]


    return InlineKeyboardMarkup(keyboard)


def create_query_buttons():
    keyboard = [
        [InlineKeyboardButton("Доуточнить запрос", callback_data="refine_query")],
        [InlineKeyboardButton("Сгенерировать новый запрос", callback_data="generate_new_query")]
    ]
    return InlineKeyboardMarkup(keyboard)