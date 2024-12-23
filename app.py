import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_API_KEY  = os.getenv("OPENAI_API_KEY")
 
#LangChainの設定
llm = ChatOpenAI(
    openai_api_key=OPEN_API_KEY,
    model="gpt-3.5-turbo", 
    temperature=0.7
)

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory)

st.title("AI Chat App with LangChain")

#入力ボックス
user_input = st.text_input("Enter your message:")

#応答を生成
if user_input:
    with st.spinner("Thinking..."):
        response = conversation.run(user_input)
    st.write(f"AI:{response}")

#会話履歴を表示
st.write("Conversation history:")
st.write(memory.buffer)
