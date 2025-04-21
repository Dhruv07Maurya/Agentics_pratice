# **********************************************************************************************************************
# Mini Project 02 : Building an intelligent MongoDB Atlas query assistant powered by LLMs
# **********************************************************************************************************************

import os  # Added to load environment variables
from langchain_groq import ChatGroq
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import faiss
import numpy as np

# Connect to LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client["sample_mflix"]
collection = db["movies"]

# print(collection.count_documents({"runtime": 11}))  # Example query to count documents with runtime = 11
# print(collection.find_one({"title": "A Corner in Wheat"}))
# print(collection.count_documents({}))  # Count all documents in the collection
# x=collection.find({"title": "The Great Train Robbery"}, {"plot": 1, "_id": 0}) # Example query to find a document with title "the great train robbery" and return only the plot field

# Few-shot examples
few_shots = []

# 1️⃣ Count documents with runtime = 11
q1 = "How many movies have a runtime of exactly 11 minutes?"
mongo_query1 = {"runtime": 11}
result1 = collection.count_documents(mongo_query1)
a1 = f"There are {result1} movies with a runtime of 11 minutes."

few_shots.append({
    "Question": q1,
    "MongoQuery": 'collection.count_documents({"runtime": 11})',
    "Result": str(result1),
    "Answer": a1
})

# 2️⃣ Find one document with title = "A Corner in Wheat"
q2 = "What is the information about the movie titled 'A Corner in Wheat'?"
mongo_query2 = {"title": "A Corner in Wheat"}
result2 = collection.find_one(mongo_query2)
a2 = f"The details for 'A Corner in Wheat' are: {result2}" if result2 else "No movie found with that title."

few_shots.append({
    "Question": q2,
    "MongoQuery": 'collection.find_one({"title": "A Corner in Wheat"})',
    "Result": str(result2),
    "Answer": a2
})

# 3️⃣ Count all documents
q3 = "How many movies are there in total in the collection?"
result3 = collection.count_documents({})
a3 = f"There are {result3} total movies in the collection."

few_shots.append({
    "Question": q3,
    "MongoQuery": 'collection.count_documents({})',
    "Result": str(result3),
    "Answer": a3
})

# 4️⃣ Get plot of "The Great Train Robbery"
q4 = "What is the plot of the movie titled 'The Great Train Robbery'?"
mongo_query4 = {"title": "The Great Train Robbery"}
projection4 = {"plot": 1, "_id": 0}
result4 = collection.find_one(mongo_query4, projection4)
a4 = f"The plot of 'The Great Train Robbery' is: {result4['plot']}" if result4 and 'plot' in result4 else "Plot not found."

few_shots.append({
    "Question": q4,
    "MongoQuery": 'collection.find_one({"title": "The Great Train Robbery"}, {"plot": 1, "_id": 0})',
    "Result": str(result4),
    "Answer": a4
})

# 1️⃣ Count of Comedy Movies
q1 = "How many Comedy movies are there?"
query1 = {"genres": "Comedy"}
result1 = collection.count_documents(query1)
a1 = f"There are {result1} Comedy movies in the collection."
few_shots.append({
    "Question": q1,
    "MongoQuery": 'collection.count_documents({"genres": "Comedy"})',
    "Result": str(result1),
    "Answer": a1
})

# 2️⃣ Top 3 Action Movies
q2 = "What are the top 3 highest rated Action movies?"
query2 = {"genres": "Action", "imdb.rating": {"$ne": None}}
projection2 = {"title": 1, "imdb.rating": 1, "_id": 0}
result2 = list(collection.find(query2, projection2).sort("imdb.rating", -1).limit(3))
a2 = f"The top 3 highest rated Action movies are: {', '.join([movie['title'] for movie in result2])}"
few_shots.append({
    "Question": q2,
    "MongoQuery": '''
collection.find(
    {"genres": "Action", "imdb.rating": {"$ne": None}},
    {"title": 1, "imdb.rating": 1, "_id": 0}
).sort("imdb.rating", -1).limit(3)
''',
    "Result": str(result2),
    "Answer": a2
})

# 3️⃣ Average Runtime of Drama Movies
q3 = "What is the average runtime of all Drama movies?"
pipeline3 = [
    {"$match": {"genres": "Drama", "runtime": {"$ne": None}}},
    {"$group": {"_id": None, "avg_runtime": {"$avg": "$runtime"}}}
]
result3 = list(collection.aggregate(pipeline3))
avg_runtime = round(result3[0]['avg_runtime'], 2) if result3 else 0
a3 = f"The average runtime of Drama movies is {avg_runtime} minutes."
few_shots.append({
    "Question": q3,
    "MongoQuery": '''
collection.aggregate([  
    {"$match": {"genres": "Drama", "runtime": {"$ne": None}}},
    {"$group": {"_id": None, "avg_runtime": {"$avg": "$runtime"}}}
])
''',
    "Result": str(result3),
    "Answer": a3
})

# Vectorize examples
to_vectorize = []
for ex in few_shots:
    text = f"Question: {ex['Question']}\nMongoQuery: {ex['MongoQuery']}\nAnswer: {ex['Answer']}"
    to_vectorize.append(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(to_vectorize, convert_to_tensor=True)
embeddings_np = embeddings.cpu().detach().numpy()

# Create FAISS index
dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_np)
example_lookup = to_vectorize

# New user question
new_question = "waht are the top 3 highest rated Horror movies?"
new_emb = model.encode([new_question])
D, I = index.search(np.array(new_emb), k=2)

# Retrieve similar examples
example_prompt_template = "Question: {Question}\nMongoQuery: {MongoQuery}\nAnswer: {Answer}"
retrieved_examples = [few_shots[i] for i in I[0]]
formatted_examples = [example_prompt_template.format(**ex) for ex in retrieved_examples]
examples_string = "\n\n".join(formatted_examples)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["examples", "question"],
    template="""

*note: MongoDB query must be in PyMongo syntax and not in any other format.
*note: also the querry in the question must be copied in same case as it is in the question.
You are a MongoDB query expert. Given a natural language question, your job is to:
- Write a valid MongoDB query using PyMongo syntax.
- Return the result in plain language.

Use the format:
Question: <user question>
MongoQuery: <PyMongo-style query>
Result: <Result of executing the query>
Answer: <natural language answer>

Here are some examples:

{examples}

Now answer this new question:
Question: {question}
just give the MongoDB query only for the {question} i dont need any of the explanation just the final querry in the output .
"""
)

chain = prompt_template | llm

response = chain.invoke({"examples": examples_string, "question": new_question})

print("🧠 Response from Groq LLM:\n")
print(response.content)

query_string = response.content.replace("MongoQuery: ", "")
print("Final MongoDB Query:\n", query_string)
print(type(query_string))

result = eval(query_string)
print(result)

result_2=[]

try:
   for i in result:
       print(i)
       result_2.append(i)
       print(type(i))
except Exception as e:
    print(f"⚠️not a mongo db cursor {e}")

print("Answer:", result_2)

promtp2='''conclude the final answer in common person readable format one line and in a single sentence for the given question and its answer in proper english max 150 words only
question={question}.
answer1={result}
answer2={result_2}
and just give the final answer only in the output without any explanation or extra information.
'''

prompt_template2=PromptTemplate(
    input_variables=["question","result" "result_2"],
    template=promtp2
)
response2 = prompt_template2 | llm
response2 = response2.invoke({"question": new_question, "result": result, "result_2": result_2})
print("Final Answer:", response2.content)
print(type(response2.content))
