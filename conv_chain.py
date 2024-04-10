from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain

from prompts import QUESTION_PROMPT, SYS_PROMPT_REFINE, SYS_PROMPT


class ConvChain:
    chat_history: list

    def __init__(self, vectordb, llm_model, chain_type, k):
        self.chat_history = []

        # Define retriever
        self.retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

        # Define conversation chain
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=llm_model,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=True,
            condense_question_prompt=QUESTION_PROMPT
        )
        if chain_type == "refine":
            self.qa.combine_docs_chain.initial_llm_chain.prompt.messages[0] = SYS_PROMPT_REFINE
        else:
            self.qa.combine_docs_chain.llm_chain.prompt.messages[0] = SYS_PROMPT

    def get_response(self, query):
        streaming_callback = StreamingStdOutCallbackHandler()
        result = self.qa({"question": query, "chat_history": self.chat_history}, callbacks=[streaming_callback])
        self.chat_history.append((query, result['answer']))
        return result

    def clear_history(self):
        self.chat_history = []
