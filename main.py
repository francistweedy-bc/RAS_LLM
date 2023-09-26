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

    loader = PyPDFDirectoryLoader("./KnowledgeFiles/")
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])

    from langchain import FewShotPromptTemplate

    # create our examples
    examples = [
        {
            "query": "How are you?",
            "answer": "I can't complain but sometimes I still do."
        }, {
            "query": "What time is it?",
            "answer": "It's time to get a watch."
        }
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """The following are exerpts from conversations with an AI
    assistant. The assistant is typically sarcastic and witty, producing
    creative  and funny responses to the users questions. Here are some
    examples: 
    """
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )
    

    query ="List all the steps to hard reboot the machine."
    # Bad queries
    # query ="When do I hard reboot?" | Reason: AI ignore opinion question and says "You should hard reboot".
    # Doc does not contain information to support any opinion, answer should be "I don't know".
    print(index.query(query))




if __name__ == "__main__":
    main()
