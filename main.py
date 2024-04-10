from flask import Flask, render_template, request, session, Response
from AI_model import get_model
from conv_chain import ConvChain
from langchain.vectorstores import DocArrayInMemorySearch

from embedding import load_vectorstore

chain_type = "refine"  # "refine"  "map_reduce"   "map_rerank"  "stuff"
k = 3  # the number of splits sent to the LLM after search in vectordb

vectordb: DocArrayInMemorySearch = load_vectorstore()

GPT35_model = get_model("GPT3.5")
cc_GPT35 = ConvChain(vectordb, GPT35_model, chain_type, k)

GPT4_model = get_model("GPT4")
cc_GPT4 = ConvChain(vectordb, GPT4_model, chain_type, k)

# Vertex_model = get_model("VertexAI")
# cc_vertex = ConvChain(vectordb, Vertex_model, chain_type, k)


def set_model():
    if 'model' in session:
        llm_model = session['model']
    else:
        llm_model = "VertexAI"

    if llm_model == "GPT4":
        return cc_GPT4
    elif llm_model == "GPT3.5":
        return cc_GPT35
    # else:
    #     return cc_vertex


app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/model", methods=["POST"])
def choose_model():
    session['model'] = request.form["model"]
    return "Model set successfully."


@app.route("/get", methods=["GET", "POST"])
def chat():
    print(session['model'])
    cc = set_model()
    query = request.form["msg"]
    response = cc.get_response(query)
    full_response = response['answer'] + "\n"
    address = set([item.metadata['source'] for item in response['source_documents']])
    if len(address) == 1:
        full_response += "I recommend taking a look at the following page: "
    elif len(address) > 1:
        full_response += "I recommend taking a look at the following pages: "
    for item in address:
        full_response += " \n " + item
    return full_response


@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    cc = set_model()
    session.pop('chat_history', None)
    cc.clear_history()
    return "Chat history reset successfully."


if __name__ == '__main__':
    app.run(debug=True)
