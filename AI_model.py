import os
import vertexai
from secret_key import openapi_key
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials


def get_model(model):
    if model == "GPT4":
        os.environ['OPENAI_API_KEY'] = openapi_key
        llm_name = "gpt-4"
        return ChatOpenAI(model_name=llm_name, temperature=0, streaming=True)

    elif model == "GPT3.5":
        os.environ['OPENAI_API_KEY'] = openapi_key
        llm_name = "gpt-3.5-turbo-1106"
        return ChatOpenAI(model_name=llm_name, temperature=0, streaming=True)

    elif model == "VertexAI":
        key_path = 'glossy-waters-403811-9fc21d93671f.json'
        credentials = Credentials.from_service_account_file(
            key_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform'])

        if credentials.expired:
            credentials.refresh(Request())

        PROJECT_ID = '515564887663'
        REGION = 'us-central1'

        # initialize vertex
        vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
        return ChatVertexAI(temperature=0, streaming=True)
