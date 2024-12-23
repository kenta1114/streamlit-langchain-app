import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
OPENAI_API_MODEL = st.secrets.get("OPENAI_API_MODEL", "gpt-3.5-turbo")
OPENAI_API_TEMPERTURE = float(st.secrets.get("OPENAI_API_TEMPERTURE", 0.7))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please provide it in your environment variables.")
    st.stop()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    st.error("SERPAPI_API_KEY is not set. Please provide it in your environment variables.")
    st.stop()

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
OPENAI_API_TEMPERTURE = float(os.getenv("OPENAI_API_TEMPERTURE", 0.7))

# Define a function for SerpAPI search
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
    return results

# Test SerpAPI connection
try:
    res = serpapi_search("今日の東京の天気")
    print(res)
except Exception as e:
    st.error(f"Error connecting to SerpAPI: {e}")

# Load tools
try:
    tools = load_tools(["serpapi", "wikipedia"], serpapi_api_key=SERPAPI_API_KEY)
except Exception as e:
    st.error(f"Error loading tools: {e}")
    st.stop()

# Create an agent chain
def create_agent_chain():
    try:
        chat = ChatOpenAI(
            model_name=OPENAI_API_MODEL,
            temperature=OPENAI_API_TEMPERTURE,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
        }

        return initialize_agent(
            tools, 
            chat, 
            agent=AgentType.OPENAI_FUNCTIONS, 
            agent_kwargs=agent_kwargs, 
            memory=memory
        )
    except Exception as e:
        st.error(f"Error creating agent chain: {e}")
        st.stop()

# Initialize session state
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

st.title("LangChain Streamlit App")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.agent_chain.run(input=prompt, callbacks=[callback])
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error during agent response: {e}")
