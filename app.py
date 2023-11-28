import os
os.system('pip install tiktoken')
os.system('pip install transformers_stream_generator')

import gradio as gr
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def clear_session():
    return '', None


from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = snapshot_download('qwen/Qwen-14B-Chat', revision='v1.0.4')

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, revision='v1.0.4')

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,torch_dtype=torch.bfloat16, revision='v1.0.4').eval()

model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
model.generation_config.top_p = 0.8
model.generation_config.repetition_penalty = 1.1


def generate_chat(input: str, history = None):
    if input is None:
        input = ''
    if history is None:
        history = []
    history = history[-5:]
    gen = model.chat_stream(tokenizer, input, history=history)
    for x in gen:
        history.append((input, x))
        yield None, history
        history.pop()
    history.append((input, x))
    return None, history

block = gr.Blocks()
with block as demo:

    # 添加知识库选择下拉列表
    knowledge_base_selector = gr.Dropdown(
        choices=['公司制度', '党员学习', '其他选项'],
        label='选择知识库'
    )

    gr.Markdown("""<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-VL-Chat/repo?Revision=master&FilePath=assets/logo.jpg&View=true" style="height: 80px"/><p>""")
    gr.Markdown("""<center><font size=8>Qwen-14B-Chat Bot👾</center>""")
    gr.Markdown("""<center><font size=4>通义千问-14B（Qwen-14B） 是阿里云研发的通义千问大模型系列的140亿参数规模的模型。</center>""")

    chatbot = gr.Chatbot(lines=10,label='Qwen-14B-Chat', elem_classes="control-height")
    message = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 清除历史对话")
        sumbit = gr.Button("🚀 发送")

    #sumbit.click(generate_chat,
    #           inputs=[message, chatbot],
    #           outputs=[message, chatbot])


    # 修改按钮的 click 方法以包含知识库选择
    sumbit.click(generate_chat,
               inputs=[message, knowledge_base_selector, chatbot],
               outputs=[message, chatbot])




    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[message, chatbot],
                        queue=False)

demo.queue().launch(height=800, share=False)
