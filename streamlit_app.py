import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from transformers import pipeline

# Load environment variables
load_dotenv()
memory = MemorySaver()

# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the state graph
graph_builder = StateGraph(State)

# Streamlit UI settings
st.set_page_config(page_title="AI Assistant", layout="wide")

# CSS for chat-like UI
st.markdown(
    """
    <style>
    .user-message {
        background-color: #d1e7dd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        align-self: flex-start;
    }
    .assistant-message {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        align-self: flex-end;
    }
    .message-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .chat-input {
        width: 90%;
        padding: 10px;
        margin-top: 10px;
        border: 1px solid #ccc;
        border-radius: 10px;
    }
    .send-button {
        padding: 10px 20px;
        border: none;
        border-radius: 10px;
        background-color: #007bff;
        color: white;
        cursor: pointer;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

if __name__ == '__main__':
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Set up tools and retrievers
    loader = PyPDFLoader("proposal.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, embedding_model)
    retriever = db.as_retriever()
    pdf_tool = create_retriever_tool(
        retriever,
        "pdf_search",
        "Search for information about my research project. For any questions about my research project, you must use this tool!"
    )

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    tools = [pdf_tool, arxiv_tool, wiki_tool]

    # Initialize the ChatGroq model with tools
    llm = ChatGroq(groq_api_key=os.getenv('groq_api_key'), model_name="Gemma2-9b-It")
    llm_with_tools = llm.bind_tools(tools)

    # Define chatbot function
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build the state graph
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=memory)

    # Initialize session state for messages and input submission tracking
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "input_submitted" not in st.session_state:
        st.session_state["input_submitted"] = False

    # Set up a summarization pipeline for post-processing responses
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Function to clean and summarize chatbot responses
    def clean_and_summarize_response(responses):
        # Combine responses into one text
        combined_response = " ".join([resp["messages"][-1].content for resp in responses])
        
        # Summarize if response is long, otherwise return directly
        if len(combined_response) > 500:
            summary = summarizer(combined_response, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        else:
            return combined_response.strip()

    # Streamlit layout for conversation UI
    st.title("AI Assistant Chatbot")

    # Display the conversation history
    for message in st.session_state["messages"]:
        css_class = "user-message" if message[0] == "user" else "assistant-message"
        st.markdown(
            f'<div class="message-container"><div class="{css_class}">{message[1]}</div></div>',
            unsafe_allow_html=True,
        )

    # User input section
    user_input = st.text_input(
        "Type your message here...", 
        key="chat_input", 
        on_change=lambda: st.session_state.update({"input_submitted": True}),
    )

    if st.button("Send"):
        if user_input:
            st.session_state["messages"].append(("user", user_input))
            # Display a spinner while processing
            with st.spinner("Processing..."):
                try:
                    # Stream messages through the graph and get responses
                    events = graph.stream({"messages": st.session_state["messages"]}, {"configurable": {"thread_id": "1"}}, stream_mode="values")

                    # Clean and summarize the response
                    user_friendly_response = clean_and_summarize_response(events)

                    # Append the response to the conversation
                    if user_friendly_response:
                        st.session_state["messages"].append(("assistant", user_friendly_response))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Clear input after submission
    if st.session_state.get("input_submitted"):
        st.session_state["input_submitted"] = False
        st.rerun()
