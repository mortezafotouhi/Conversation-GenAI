from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate

prompt_template_refine = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the question is unrelated to the context, respond that I am unable to provide an answer to such question.

{context_str}
"""

SYS_PROMPT_REFINE = SystemMessagePromptTemplate.from_template(prompt_template_refine)

prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the question is unrelated to the context, respond that I am unable to provide an answer to such question.

{context}
"""

SYS_PROMPT = SystemMessagePromptTemplate.from_template(prompt_template)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Return the question in English. If the question is unrelated to the Chat History, just return the original version of the question. 

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
QUESTION_PROMPT = PromptTemplate.from_template(_template)
