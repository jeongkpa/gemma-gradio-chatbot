# Ollama 모델 로드 및 테스트

# langchain_community 패키지에서 ChatOllama 모델을 불러옵니다.
from langchain_community.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# gradio 패키지에서 gr 모듈을 불러옵니다.
import gradio as gr

# ChatOllama 모델을 불러와서 model 변수에 할당합니다.
# "gemma:2b-instruct" 모델을 사용하고, temperature를 0으로 설정합니다.
model = ChatOllama(model="gemma:2b-instruct", temperature=0)

def response(message, history, additional_input_info):
        history_langchain_format = []
        # additional_input_info로 받은 시스템 프롬프트를 랭체인에게 전달할 메시지에 포함시킨다.
        history_langchain_format.append(SystemMessage(content= additional_input_info))
        for human, ai in history:
                history_langchain_format.append(HumanMessage(content=human))
                history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = model(history_langchain_format)
        return gpt_response.content

gr.ChatInterface(
        fn=response,
        textbox=gr.Textbox(placeholder="질문해주세요", container=False, scale=7),
        # 채팅창의 크기를 조절한다.
        chatbot=gr.Chatbot(height=1000),
        title="어떤 챗봇을 원하심미까?",
        description="물어보면 답하는 챗봇임미다.",
        theme="soft",
        retry_btn="다시보내기 ↩",
        undo_btn="이전챗 삭제 ❌",
        clear_btn="전챗 삭제 💫",
        additional_inputs=[
            gr.Textbox("", label="System Prompt를 입력해주세요", placeholder="")
        ]
).launch()