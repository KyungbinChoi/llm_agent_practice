# %%
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')

from langchain_chroma import Chroma

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name = 'income_tax_collection',
    persist_directory = './income_tax_collection'
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document] # 기존 vector store 에서 검색한 결과와 web 검색 결과가 추가됨
    answer: str
    
graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState) -> AgentState:
    query = state['query']
    docs = retriever.invoke(query)
    return {'context':docs}

# %%
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = 'gpt-4o')

# %%
from langchain import hub
generate_prompt = hub.pull('rlm/rag-prompt')
generate_llm = ChatOpenAI(model = 'gpt-4o', max_completion_tokens=100)

def generate(state: AgentState):
    context = state['context']
    query = state['query']
    rag_chain = generate_prompt | generate_llm
    response = rag_chain.invoke({'question': query , 'context': context})
    return {'answer':response.content} # AImessage 부분을 포함시키 않기 위함

# %%
# END 로 끝내야함
from typing import Literal
relevance_doc_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant','irrelevant']:
    query = state['query']
    context = state['context']
    # print(f'context == {context}')
    
    doc_relevance_chain = relevance_doc_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    # print(f'doc relevance response : {response}')
    if response['Score']==1:
        return 'relevant'
    
    return 'irrelevant'

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']
rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
사전: {dictionary}
질문: {{query}}                             
""")

def rewrite(state: AgentState):
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query':query})
    return {'query' : response}

# %%
# from langchain import hub
# hallucination_prompt = hub.pull("langchain-ai/rag-answer-hallucination")

from langchain_core.prompts import PromptTemplate
hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal["hallucinated", "not hallucinated"]:
    answer = state['answer']
    context = state['context']
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    # hallucination_chain = hallucination_prompt | hallucination_llm 
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})

    return response

# %%
from langchain import hub
helpfulness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state:AgentState) -> Literal["helpful","unhelpful"]:
    query = state['answer']
    answer = state['answer']
    # hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    helpfulness_chain = helpfulness_prompt | llm 
    response = helpfulness_chain.invoke({'student_answer': answer, 'question': query})
    if response['Score']==1:
        return 'helpful'
    return 'unhelpful'

def check_helpfulness(state: AgentState):
    return state

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
# graph_builder.add_node('check_doc_relevance', check_doc_relevance)
graph_builder.add_node('rewrite', rewrite)
# graph_builder.add_node('check_hallucination', check_hallucination)
graph_builder.add_node('check_helpfulness', check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START , 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant' : 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'hallucinated' : 'generate',
        'not hallucinated': 'check_helpfulness'
    }
)

graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful' : END,
        'unhelpful': 'rewrite'
    }
)

graph_builder.add_edge('rewrite', 'retrieve')


# %%
graph = graph_builder.compile()
