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

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
OPENAI_API_TEMPERTURE = float(os.getenv("OPENAI_API_TEMPERTURE", 0.7))

# Ensure all required environment variables are set
required_env_vars = ["OPENAI_API_KEY", "SERPAPI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"The following environment variables are missing: {', '.join(missing_vars)}. Please add them to your .env file or environment settings.")
    st.stop()

def serpapi_search(query):
    """Search using SerpAPI."""
    try:
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
    except Exception as e:
        st.error(f"Error connecting to SerpAPI: {e}. Check your API key and network connection.")
        return None

# Example: Test SerpAPI
res = serpapi_search("今日の東京の天気")
if res:
    print(res)

# Function to create the agent chain
def create_agent_chain():
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

    try:
        tools = load_tools(["serpapi", "wikipedia"], serpapi_api_key=SERPAPI_API_KEY)
    except Exception as e:
        st.error(f"Error loading tools: {e}. Ensure SERPAPI_API_KEY is correct and that the tools are available.")
        st.stop()

    return initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory)

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

st.title("langchain-streamlit-app")

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

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())

        try:
            agent_chain = st.session_state.agent_chain
            response = agent_chain.run(prompt, callbacks=[callback])
            st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error during agent response: {e}")
