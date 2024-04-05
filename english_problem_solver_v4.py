# English Problem Solver v3
# 기능1. 텍스트 입력하면 gpt-4-turbo가 답변

# 기능2. 이미지 업로드하면 gpt-4-vision이 답변
# - 그래프 이미지 업로드
# - 지문 텍스트 복사붙여넣기

# share streamlit 에서 배포

import base64
from PIL import Image as Img
import time

import streamlit as st
import openai
from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
# openai.api_key = os.environ.get('OPENAI_API_KEY')

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

from langchain.chat_models import ChatOpenAI
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

st.header("📄English Problem Solver📊")

# 사이드에 list의 선택박스를 생성
select = st.sidebar.selectbox('메뉴', ['그래프 이미지 분석', '텍스트 분석'])

if select == '그래프 이미지 분석':

    # 그래프 이미지 업로드
    st.text("")
    st.text("")
    graph_image_file = st.file_uploader("그래프 이미지를 입력해주세요", type=['png', 'jpg', 'jpeg'], key="graph_image")
    if graph_image_file is not None:
        st.image(graph_image_file, width=500)
    else:
        st.info('☝️ 그래프 문제를 업로드하면 문제를 풀어드립니다.')


    # 지문 텍스트 업로드
    st.text("")
    st.text("")
    passage = st.text_input("지문 텍스트를 입력해주세요(입력 후 엔터)")


    # # 문제 전체 이미지 입력
    # st.text("")
    # st.text("")
    # full_image_file = st.file_uploader("전체 문제 이미지를 입력해주세요", type=['png', 'jpg', 'jpeg'], key="full_image")

    # 버튼 누르면 gpt에게 답변 받기 시작
    if st.button('풀어줘!'):
        start = time.time()  # 시작 시간 저장

        # --------------------------deplot으로 수치 추출하기-------------------------

        processor = Pix2StructProcessor.from_pretrained('nuua/ko-deplot')
        model = Pix2StructForConditionalGeneration.from_pretrained('nuua/ko-deplot')
        
        image = Img.open(graph_image_file)
        inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
        predictions = model.generate(**inputs, max_new_tokens=512)
        result = processor.decode(predictions[0], skip_special_tokens=True)
        # result = ""

        # st.success(result)

        # --------------------------gpt-4-vision으로 문제 풀기--------------------------

        # 메시지 히스토리 생성
        messages=[{"role": "system", "content": """You must answer in Korean. You're a mathematician who calculates numbers and makes comparisons with rigor and precision.
                    For comparing numbers, compare based on the absolute value of the subtraction.
                    If about the largest or smallest numbers, compare the numbers across all groups.
                    """}]

        # 첫번째 질문 생성
        question = f""" 그래프 분석 result를 제공해줄게. 
        graph analysis : {result}. 
        Q. 그래프의 내용과 가장 맞지 않을 확률이 높은 문장은?
        {passage}. 
        """ 

        graph_image_file = graph_image_file.name

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
        # Getting the base64 string
        base64_image = encode_image(graph_image_file)

        question_dict = {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                    },
                ],
                }

        # 메시지 히스토리에 질문 추가
        messages.append(question_dict)

        # 답변 받기
        completion = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages = messages,
        )

        response = completion.choices[0].message.content

        st.info(response)

        st.write("추론 시간 :", int(time.time() - start), "초")     # 현재시각 - 시작시간 = 실행 시간


if select == '텍스트 분석':
    # 사용자 질문 입력
    question = st.text_area("문제를 입력해주세요", height=10)

    # 메시지 히스토리 생성
    messages=[{"role": "system", "content": """
               You must answer in Korean. You're a mathematician who calculates numbers and makes comparisons with rigor and precision.
               Think in steps when making size comparisons between figures. 
               """}]

    question_dict = {
                "role": "user",
                "content": question ,  # 첫 번째 질문
            }

    # 메시지 히스토리에 질문 추가
    messages.append(question_dict)

    # 버튼 누르면 답변 받기 시작
    if st.button('풀어줘!'):
    # st.write('Why hello there')

        # 답변 받기
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",          # "gpt-3.5-turbo-0125", "gpt-4-0125-preview"
            messages = messages,
        )

        response = completion.choices[0].message.content

        st.info(response)
