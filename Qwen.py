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



default_system = '你是国网湖北省电力有限公司的秘书，会根据用户的要求和材料，写出专业、详实、高质量的文本段落。文章需要有创造力，不拘泥于参考的文本，并且可以适当扩展，尽量写的详细。'

dashscope.api_key = "**"


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
        query = query_process(query,Thread_value=0.6,max_return=2)
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


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>Qwen Chat Bot👾</center>""")

    with gr.Row():
        with gr.Column(scale=3):
            system_input = gr.Textbox(value=default_system, lines=1, label='System')
        with gr.Column(scale=1):
            modify_system = gr.Button("🛠️ 设置system并清除历史对话", scale=2)
        system_state = gr.Textbox(value=default_system, visible=False)
    chatbot = gr.Chatbot(label='Qwen-72B-Chat')
    textbox = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        sumbit = gr.Button("🚀 发送")
        # regen_btn = gr.Button("🤔️ Regenerate (重试)")
        clear_history = gr.Button("🧹 清除历史对话")


    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_state],
                 outputs=[textbox, chatbot, system_input])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot])
    modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot])

AUTHENTICATION = [("demo", "demo123")]
demo.queue(concurrency_count=10,api_open=False).launch(
        height=800,
        share=True,
        # inbrowser=False,
        server_port=11823,
        server_name='0.0.0.0',
    auth=AUTHENTICATION
    )