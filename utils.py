import re

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