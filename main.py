from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader


from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator

import os

def main():
    api_key = ""
    with open("api_key.txt", "r") as f:
        api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = OpenAI(temperature=0.9)

    # prompt = f"""You are a tech support expert that knows everything about the machine mentioned in the documents: {index}.\
    #             Advise on the best way to solve the problems mentioned in the question delimited by the triple backticks.\
    #             question: ```{question}```
    #     into a style that is {style}.
    #     text: ```{customer_email}```
    #     """

    loader = PyPDFDirectoryLoader("./KnowledgeFiles/")
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    

    # query = "How do I do a soft reboot?"
    # query ="How do I turn the machine off?"
    query ="List all the steps to hard reboot the machine."
    # query ="How do I restart the machine?"
    # query ="My machine is not working."
    # query ="Hello, how are you today?"
    # query ="What is 2+2?"

    # Bad queries
    # query ="When do I hard reboot?" | Reason: AI ignore opinion question and says "You should hard reboot".
    # Doc does not contain information to support any opinion, answer should be "I don't know".
    print(index.query(query))




if __name__ == "__main__":
    main()
