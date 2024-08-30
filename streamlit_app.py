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
import tempfile

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
        background-color: #1F2060;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        align-self: flex-start;
    }
    .assistant-message {
        background-color: #2B601F;
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
    # Streamlit layout for conversation UI
    st.title("AI Assistant Chatbot")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    # Add file uploader for the user to upload multiple PDF documents
    uploaded_files = st.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)

    # Initialize an empty list to hold all documents
    all_documents = []

    # Check if a file is uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Load each uploaded file as a document
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            documents = text_splitter.split_documents(docs)
            all_documents.append(uploaded_file.name)
            all_documents.extend(documents)

        # Create FAISS database from all combined documents
        db = FAISS.from_documents(all_documents, embedding_model,)
        retriever = db.as_retriever()
        pdf_tool = create_retriever_tool(
            retriever,
            "pdf_search",
            "Search for information about proposal of my research project. For any questions about my research project, you must use this tool!"
        )
        tools = [pdf_tool, arxiv_tool, wiki_tool]
    else:
        st.warning("Please upload a PDF document to continue.")
        tools = [arxiv_tool, wiki_tool]  # Exclude PDF tool if no document is uploaded


    # Initialize the ChatGroq model with tools
    llm = ChatGroq(groq_api_key=os.getenv('groq_api_key'), model_name="llama-3.1-70b-versatile")
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

    # Display the conversation history
    for message in st.session_state["messages"]:
        css_class = "user-message" if message[0] == "user" else "assistant-message"
        st.markdown(
            f'<div class="message-container"><div class="{css_class}">{message[1]}</div></div>',
            unsafe_allow_html=True,
        )

    # Chat input section - keeping input field static at the bottom
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state["messages"].append(("user", user_input))
        # Stream messages through the graph and get responses
        events = graph.stream({"messages": st.session_state["messages"]}, {"configurable": {"thread_id": "1"}}, stream_mode="values")

        # Collect and combine responses
        combined_response = []
        for event in events:
            response = event["messages"][-1].content
            if response.strip():
                combined_response.append(response)

        for event in events:
            event["messages"][-1].pretty_print()

        # Append the response to the conversation
        if len(combined_response) > 0:
            st.session_state["messages"].append(("assistant", combined_response[-1]))

        # Rerun the app to clear the input after submission
        st.rerun()
