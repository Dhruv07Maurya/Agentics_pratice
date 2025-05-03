# **********************************************************************************************************************
# BASIC CONFIGURATION FOR LANGCHAIN GROQ API KEY AND MODEL NAME
# **********************************************************************************************************************

# from langchain_groq import ChatGroq
# import os
# llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
# response = llm.invoke("What is langchain")
# print(response.content)
# # os.environ["GOOGLE_API_KEY"]
# prompt1=PromptTemplate(
#     input_variables=["topic"],
#     template="What is {topic} in 20 words"
# )
# chain1=LLMChain(llm=llm, prompt=prompt1, output_key="paragraph")
# prompt2=PromptTemplate(
#     input_variables=["paragraph"],
#     template="summarize {paragraph} in 5 words"
# )
# chain2=LLMChain(llm=llm, prompt=prompt2, output_key="summary")
# chain=SequentialChain(
#     chains=[chain1, chain2],
#     input_variables=["topic"],
#     output_variables=["paragraph", "summary"]
# )

# **********************************************************************************************************************
# CHAINS TOOLS AND AGENTS INITIALIZATION pratice
# **********************************************************************************************************************

# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, SequentialChain
# from langchain_groq import ChatGroq
# from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain.agents import initialize_agent, AgentType
# import os

# llm=ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

# tools=load_tools(["wikipedia"], llm=llm)

# agent=initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# response=agent.run("When was elon musk born and what is his age today?")
# print(response)


# LOADING DATA FROM URLS AND TEXT FILES AND SPLITTING INTO CHUNKS 

# from langchain.document_loaders import TextLoader
# loader = TextLoader("data.txt", encoding='utf-8')
# print(loader.load()[0].metadata)

# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# loader=UnstructuredURLLoader(
#         urls=["https://www.ndtv.com/india-news/did-anyone-die-lamborghini-driver-after-hitting-2-workers-on-noida-footpath-8047514#pfrom=home-ndtv_topscroll"]
#     )
# data=loader.load()
# para=data[0].page_content

# splitter=RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", "."],
#     chunk_size=200,
#     chunk_overlap=50,
# )
# chunks=splitter.split_text(para)
# print(len(chunks))

# print(chunks[0])

# for chunk in chunks:
#     print(len(chunk))



# **********************************************************************************************************************
# VECTOR EMBEDDINGS AND FAISS INDEXING 
# **********************************************************************************************************************

# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# sentences = ["king", "queen","man", "woman"]

# model = SentenceTransformer('all-MiniLM-L6-v2')
# vectors = model.encode(sentences)
# dim= vectors.shape[1]
# print(dim)
# print(vectors.shape)
# index = faiss.IndexFlatL2(dim)
# print(index)
# index.add(vectors)

# search_querry= "King minus Man plus Woman equals to"
# search_vector = model.encode(search_querry)
# print(search_vector.shape)

# s_vec= np.array([search_vector])
# print(s_vec.shape)

# distances, indices = index.search(s_vec, k=1)
# print("Distances:", distances)
# print("Indices:", indices)
# print("Search Results:")
# for idx in indices[0]:
#     print(sentences[idx])


# **********************************************************************************************************************
# MINI PROJECT 01 : TO BUILD A CHATBOT USING FAISS INDEXING AND EMBEDDINGS WHICH CAN ANSWER QUESTIONS RELATED TO SPECIFIC URLS 
# **********************************************************************************************************************


# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import faiss
# import numpy as np
# import os

# llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

# response = UnstructuredURLLoader(
#     urls=["https://edition.cnn.com/business/live-news/tariffs-trump-news-04-02-25/index.html"]
# )
# data = response.load()[0]

# chunks = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n",],
#     chunk_size=1500,
#     chunk_overlap=300,
# )
# chunks = chunks.split_text(data.page_content)
# print(len(chunks))

# model = SentenceTransformer('all-MiniLM-L6-v2')
# vectors = model.encode(chunks)
# print(vectors.shape)
# dim = vectors.shape[1]
# index = faiss.IndexFlatL2(dim)
# print(index)
# index.add(vectors)

# question = input("Ask me a Question: ")

# search_vector = model.encode(question)
# s_vec = np.array([search_vector])

# distances, indices = index.search(s_vec, k=3)

# prompt_template = """
# You are an assistant that helps answer questions using the provided information. 
# Answer the question based on the following content from a document:

# strictness: if the question is not related to the content,strictly say "I don't know".

# {chunks}

# Question: {question}
# Answer:
# """

# prompt = PromptTemplate(input_variables=["chunks", "question"], template=prompt_template)
# chain = LLMChain(llm=llm, prompt=prompt)

# relevant_chunks = "\n\n".join([chunks[idx] for idx in indices[0]])

# answer = chain.run({"chunks": relevant_chunks, "question": question})

# print("\nAnswer from LLM:\n", answer)

# Optionally, you can print the relevant chunks for clarity
# print("\nRelevant Chunks used for Answer:")
# for idx in indices[0]:
#     print(chunks[idx])


# **********************************************************************************************************************
# Mini Project 02 : Building an intelligent MongoDB Atlas query assistant powered by LLMs
# **********************************************************************************************************************


# prompt1=PromptTemplate(
#     input_variables=["topic"],
#     template="What is {topic} in 20 words"
# )
# chain1=LLMChain(llm=llm, prompt=prompt1, output_key="paragraph")

# import os
# llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
# response = llm.invoke("What is langchain in 20 words")
# print(response.content)


# **********************************************************************************************************************
# voice based model calling

# const groqApiKey =
# process.env.GROQ_API_KEY;
# const elevenLabsApiKey =
# process.env.ELEVEN_LABS_API_KEY;
# const voiceID = "EXAMPLE_VOICE_ID";
# print(groq)
