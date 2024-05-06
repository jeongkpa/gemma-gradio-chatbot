# Ollama 모델 로드 및 테스트

# langchain_community 패키지에서 ChatOllama 모델을 불러옵니다.
from langchain_community.chat_models import ChatOllama

# gradio 패키지에서 gr 모듈을 불러옵니다.
import gradio as gr

# ChatOllama 모델을 불러와서 model 변수에 할당합니다.
# "gemma:2b-instruct" 모델을 사용하고, temperature를 0으로 설정합니다.
model = ChatOllama(model="gemma:2b-instruct", temperature=0)

# 사용자 정의 함수를 정의합니다.
# 이 함수는 사용자가 입력한 메시지와 이전 대화 내용을 받아와서 모델을 사용하여 응답을 생성합니다.
def echo(message, history):
    # 모델을 사용하여 입력된 메시지에 대한 응답을 생성합니다.
    response = model.invoke(message)
    # 생성된 응답을 반환합니다.
    return response.content

# gradio를 사용하여 대화 인터페이스를 만듭니다.
# 사용자 정의 함수인 echo를 기능으로 사용하고, 인터페이스의 제목을 "Gemma:2b-instruct Bot"으로 설정합니다.
demo = gr.ChatInterface(fn=echo, title="Gemma:2b-instruct Bot")

# 대화 인터페이스를 실행합니다.
demo.launch()
