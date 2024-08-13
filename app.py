import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import requests
import json

from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Custom text splitter for summarization
def custom_text_splitter(text, chunk_size=1000, chunk_overlap=20):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def summarize(text):
    chunks = custom_text_splitter(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    prompt_template = """Write a concise summary of the following text:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    llm = ChatOpenAI(temperature=0.7)
    summarize_chain = LLMChain(llm=llm, prompt=prompt)
    
    summaries = []
    for doc in docs:
        summary = summarize_chain.run(doc.page_content)
        summaries.append(summary)
    
    return " ".join(summaries)


def web_search(query: str) -> str:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    
    if response.status_code == 200:
        results = response.json()
        formatted_results = []
        for item in results.get('organic', [])[:3]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No snippet')
            link = item.get('link', 'No link')
            formatted_results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
        return "\n".join(formatted_results)
    else:
        return f"Error in web search: {response.status_code} - {response.text}"
    
tools = [
    Tool(
        name="Web Search",
        func=web_search,
        description="Search the web for current information.",
    ),
    Tool(
        name="Summarizer",
        func=summarize,
        description="Summarize long pieces of text.",
    )
]


# Set up the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize the agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

@app.route("/query", methods=["POST"])
def query_agent():
    data = request.json
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data["question"]
    try:
        response = agent.run(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
