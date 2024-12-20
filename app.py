import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import MessagesPlaceholder
from serpapi import GoogleSearch
from langchain.llms import OpenAI

llm = ChatOpenAI(temperature=0.7)

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
OPENAI_API_TEMPERTURE=float(os.getenv("OPENAI_API_TEMPERTURE",0.7))

def serpapi_search(query):
    params = {
        "engine": "google",
        "q": query,
        "hl": "ja",
        "gl": "jp",
        "api_key": SERPAPI_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    # 必要に応じてresultsからタイトルやスニペットを抽出
    return results

# 例: 東京の今日の天気を検索
res = serpapi_search("今日の東京の天気")
print(res)

tools = load_tools(["serpapi","wikipedia"])

agent = initialize_agent(tools,llm, agent="zero-shot-react-description",verbose=True)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def create_agent_chain():
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=float(os.environ.get("OPENAI_API_TEMPERTURE",0.7)),
        streaming=True,
    )

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    agent_kwargs={
        "extra_prompt_messages":[MessagesPlaceholder(variable_name="chat_history")],
    }

    tools = load_tools(["serpapi","wikipedia"])
    return initialize_agent(tools,chat,agent=AgentType.OPENAI_FUNCTIONS,agent_kwargs=agent_kwargs,memory=memory)

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain=create_agent_chain()

st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # chat = ChatOpenAI(
        #     model_name=os.environ["OPENAI_API_MODEL"],
        #     temperature=os.environ["OPENAI_API_TEMPERTURE"],
        # )
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = st.session_state.agent_chain

        response = agent_chain.run(prompt,callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant","content":response})