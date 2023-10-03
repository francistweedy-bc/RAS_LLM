from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain, RetrievalQA
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# fix the sqlite3 version issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os

def main():
    api_key = ""
    with open("api_key.txt", "r") as f:
        api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = OpenAI(temperature=0.9)

    # load PDFs
    loader = PyPDFDirectoryLoader("./KnowledgeFiles/")
    documents = loader.load()

    # split PDFs into pages
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # create vector store with split documents
    vectordb = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
        persist_directory='./data'
    )
    vectordb.persist()

    # create chain for the query
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
        return_source_documents=True
    )

    # we can now execute queries against our Q&A chain
    result = qa_chain({'query': 'List all the steps to hard reboot the machine.'})
    print(result['result'])


if __name__ == "__main__":
    main()
