import os
import gradio as gr
from http import HTTPStatus
import dashscope
import numpy as np
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from typing import List, Optional, Tuple, Dict
from urllib.error import HTTPError
import os
from get_text_embedding import embed_text,query_process
import pandas as pd

default_system = '你是国网湖北省电力有限公司的秘书，会根据用户的要求和材料，写出专业、详实、高质量的文本段落。文章需要有创造力，不拘泥于参考的文本，并且可以适当扩展，尽量写的详细专业。'

dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = dashscope_api_key






History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

def clear_session() -> History:
    return '', []

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []

def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history




def model_chat(query: Optional[str], history: Optional[History], system: str
) -> Tuple[str, str, History]:
    if query is None:
        query = ''
    else:
        query = query_process(query,Thread_value=0.5,max_return=2)
    if history is None:
        history = []
    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    gen = Generation.call(
        model = "qwen-72b-chat",
        messages=messages,
        result_format='message',
        stream=True,
        max_tokens=1500,

    )
    for response in gen:
        if response.status_code == HTTPStatus.OK:
            role = response.output.choices[0].message.role
            response = response.output.choices[0].message.content
            system, history = messages_to_history(messages + [{'role': role, 'content': response}])
            yield '', history, system
        else:
            error_message = 'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            )
            raise Exception(error_message)

if __name__ == "__main__":
    if __name__ == "__main__":
        system = default_system
        history = []

        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
            try:
                for _, history, system in model_chat(user_input, history, system):
                    for user_message, assistant_message in history:
                        print(f"User: {user_message}")
                        print(f"Assistant: {assistant_message}")
            except Exception as e:
                print("An error occurred:", e)

