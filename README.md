# AI Agent API

This project implements a simple AI agent API using Flask, LangChain, and OpenAI's GPT-3.5-turbo model. The API provides a single endpoint that allows users to ask questions, which are then processed by an AI agent capable of performing calculations.

## Features

- Flask-based API
- LangChain integration for AI agent setup
- OpenAI's GPT-3.5-turbo model for natural language processing
- Simple calculator tool for mathematical operations

## Requirements

- Python 3.7+
- Flask
- python-dotenv
- langchain
- openai

## Installation

1. Clone this repository

2. Create Python Virtual Environment
   
    python -m venv mvenv

3. Acivate the Python Virtual Environment
   
   source venv/bin/activate  # On Windows, use `mvenv\Scripts\activate`

4. Install the required packages:
   
   pip install -r requirements.txt

5. Set up your OpenAI API key in a `.env` file (see Configuration section)

## Configuration
    Create a `.env` file in the root directory of the project and add your OpenAI API key:
    OPENAI_API_KEY=your_api_key_here
    SERPER_API_KEY=your_serper_api_key_here

## Usage

1. Run the Flask application:
    python app.py

2. To query the AI agent, send a POST request to the `/query` endpoint:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"question":"What are the latest developments in AI?"}' http://127.0.0.1:5000/query

3. The API will return a JSON response with the agent's answer.
