import pickle
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


def load_vectorstore():
    # load splits
    with open('splits_nhs.pkl', 'rb') as splits_file:
        splits = pickle.load(splits_file)

    # embedding
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embedding_function)

    return vectordb
