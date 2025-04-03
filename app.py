import gradio as gr
import os
from groq import Groq
import pandas as pd
from datasets import Dataset
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

embeds = encoder(["this is a test"])
dims = len(embeds[0])

############ TESTING ############

import os
import getpass
from pinecone import Pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

import time

index_name = "groq-llama-3-rag"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=dims,
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()


def get_docs(query: str, top_k: int) -> list[str]:
    # encode query
    xq = encoder([query])
    # search pinecone index
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    # get doc text
    docs = [x["metadata"]['content_snippet'] for x in res["matches"]]
    return docs

from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate(query: str, history):

    # Create system message
    if not history:
        system_message = (
            "You are a friendly and knowledgeable New Yorker who loves sharing recommendations about the city. "
            "You have lived in NYC for years and know both the famous tourist spots and hidden local gems. "
            "Your goal is to give recommendations tailored to what the user is asking for, whether they want iconic attractions "
            "or lesser-known spots loved by locals.\n\n"
            "Use the provided context to enhance your responses with real local insights, but only include details that are relevant "
            "to the user’s question. If the context provides useful recommendations that match what the user is asking for, use them. "
            "If the context is unrelated or does not fully answer the question, rely on your general NYC knowledge instead.\n\n"
            "Be specific when recommending places—mention neighborhoods, the atmosphere, and why someone might like a spot. "
            "Keep your tone warm, conversational, and engaging, like a close friend who genuinely enjoys sharing their city.\n\n"
            "CONTEXT:\n"
            "\n---\n".join(get_docs(query, top_k=5))
        )
        messages = [
            {"role": "system", "content": system_message},
        ]
    else:
        # Establish history
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        system_message = (
            "Here is additional context based on the newest query.\n\n"
            "CONTEXT:\n"
            "\n---\n".join(get_docs(query, top_k=5))
        )
        messages.append({"role": "system", "content": system_message})

    # Add query
    messages.append({"role": "user", "content": query})
    
    # generate response
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content


# Custom CSS for iPhone-style chat
custom_css = """
.gradio-container {
    background: transparent !important;
}
.chat-message {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.chat-message.user {
    justify-content: flex-end;
}
.chat-message.assistant {
    justify-content: flex-start;
}
.chat-bubble {
    padding: 10px 15px;
    border-radius: 20px;
    max-width: 70%;
    font-size: 16px;
    display: inline-block;
}
.chat-bubble.user {
    background-color: #007aff;
    color: white;
    border-bottom-right-radius: 5px;
}
.chat-bubble.assistant {
    background-color: #f0f0f0;
    color: black;
    border-bottom-left-radius: 5px;
}
.profile-pic {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin: 0 10px;
}
"""

# Gradio Interface
demo = gr.ChatInterface(generate, css=custom_css)

demo.launch()
