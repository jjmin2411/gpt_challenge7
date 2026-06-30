from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st


answers_prompt = ChatPromptTemplate.from_template(
"""
Using ONLY the following context answer the user's question.
If you can't just say you don't know, don't make anything up.

Then, give a score to the answer between 0 and 5.
The score should be high if the answer is related to the user's question, and low otherwise.
If there is no relevant content, the score is 0.
Always provide scores with your answers
Context: {context}

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

Your turn!
Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt |llm
    return{ "answers": [
       {
           "answer": answers_chain.invoke(
               {"question": question, "context": doc.page_content}
           ).content,
           "source": doc.metadata["source"],
           "date": doc.metadata["lastmod"],
       } for doc in docs
    ],
    "question": question,
    }

choose_prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
Use ONLY the following pre-existing answers to answer the user's question.
Use the answers that have the highest score (more helpful) and favor the most recent ones.
Cite sources and return the sources of the answers as they are, do not change them.
Answers: {answers}
""",
),
("human", "{question}"),
]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"for answer in answers)
    return choose_chain.invoke({
        "question": question,
        "answers": condensed
    })
  


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
    url,
    filter_urls=[
    r"^https://developers\.cloudflare\.com/ai-gateway/.*",
    r"^https://developers\.cloudflare\.com/vectorize/.*",
    r"^https://developers\.cloudflare\.com/workers-ai/.*",
    ],
    parsing_function=parse_page,
    )   
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        chunk_size=100,
    )
    vector_store = FAISS.from_documents(docs,embeddings)
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    st.link_button("GitHub Repository", "https://github.com/jjmin2411/gpt_challenge7")
    api_key = st.text_input("OpenAI API Key", type="password")

url = "https://developers.cloudflare.com/sitemap-0.xml"

if not api_key:
    st.warning("Please provide your OpenAI API Key.")
else:
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key,
    )

    retriever = load_website(url, api_key)
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
        result = chain.invoke(query)
        st.write(result.content)

