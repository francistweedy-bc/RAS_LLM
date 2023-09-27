from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain import FewShotPromptTemplate, PromptTemplate
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

    # START of new code
    # create our examples
    examples = [
        {
            "query": "How do I remove the scanning tower?",
            "answer": "I'm sorry, I don't have the information for that."
        }, {
            "query": "When do I hard reboot?",
            "answer": "I'm sorry, I don't have the information for that."
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
    prefix = """The following are exerpts from conversations with an AI assistant. 
    The AI is qualified to answer knowledge questions, but not questions that want an opinion.  
    If the ai doesn't know the answer or gets a question that wants an opinion, it will say 'I'm sorry, I don't have the information for that.' 
    Here are some examples: 
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
    # END of new code
    

    query ="List all the steps to hard reboot the machine."
    # Bad queries
    # query ="When do I hard reboot?" | Reason: AI ignore opinion question and says "You should hard reboot".
    # Doc does not contain information to support any opinion, answer should be "I don't know".
    # print(index.query(query))
    print(few_shot_prompt_template.format(query=query))




if __name__ == "__main__":
    main()
