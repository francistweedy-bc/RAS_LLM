from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
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
            "query": "What is your name?",
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
    prefix = """Use the following context to answer the question at the end.  If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        Question: {query}
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
        input_variables=["context", "query"],
        example_separator="\n\n"
    )
    # END of new code
    chain = load_qa_chain(llm, chain_type="stuff", prompt=few_shot_prompt_template)
    chain({"input_documents": index, "query": "What is your name?"}, return_only_outputs=True)
    
    # chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)
    # chain = load_qa_chain(llm, chain_type="stuff", prompt=few_shot_prompt_template)
    # print(chain({"input_documents": index, "question": "List all the steps to hard reboot the machine."}, return_only_outputs=True))


    # Bad queries
    # query ="When do I hard reboot?" | Reason: AI ignore opinion question and says "You should hard reboot".
    # Doc does not contain information to support any opinion, answer should be "I don't know".
    # print(index.query(query))
    # print(few_shot_prompt_template.format(query=query))
    # print(chain.run("List all the steps to hard reboot the machine."))
    # print(chain.run("When do I do a hard reboot?"))




if __name__ == "__main__":
    main()
