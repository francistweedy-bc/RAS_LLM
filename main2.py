import sys
import os
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain, RetrievalQA
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app_token = "app token"
bot_token = "bot token"
app = App(token=bot_token)


# fix the sqlite3 version issue
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# def main():
@app.event("app_mention")
def mention_handler(body, say):
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

    # create vector store(embeddings) with split documents
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
        return_source_documents=True,
    )

    # get the Slack thread and user question from message
    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    question = body.get("text")

    # pass Slack question to LLM chain and get result and source documents
    result = qa_chain({'query': question})
    doc_sources = set()
    for x in result["source_documents"]:
        doc_sources.add(x.metadata["source"])

    # return result and source documents to Slack user
    new_line = '\n'
    say(test=f'{result}{new_line}{doc_sources}', thread_ts=thread_ts)    
    # print(result['result'])
    # print(doc_sources)


if __name__ == "__main__":
    # main()
    handler = SocketModeHandler(app, app_token).start()
    handler.start()
