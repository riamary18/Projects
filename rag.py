from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Open text -> read content -> chunking(splitter) -> get embeddings -> create vectore store (ids, chunks, embeddings) -> 
# retrieve top relevant chunks(question, collection) -> get llm -> generate answer

def read_contents(file):
    with open(f'{file}.txt', 'r') as f:
        text = f.read()
    return text

def chunking(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks, model):
    embeddings = model.embed_documents(chunks)
    return embeddings

def store_vec_embeddings(chunks, embeddings):
    client = chromadb.Client()
    collection = client.create_collection("novatech")
    chunk_ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=chunk_ids, documents=chunks,embeddings=embeddings)
    return collection

def retrieve(model, question, collection):
    question_embedding = model.embed_query(question)
    results = collection.query(query_embeddings = [question_embedding], n_results = 3)
    return results

def get_llm():
    load_dotenv()
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def generate_answer(llm, question, context):
    prompt = f"""You are a question-answering assistant.
    Given the context, answer the question given to you, taking content ONLY from the context.
    If the answer cannot be found in the context, do not provide your own answer.
    Return 'Text doesn't contain your question's answer.'
    Question: {question}
    Context:{context}"""
    answer = llm.invoke(prompt)
    return answer

text = read_contents("NovaTech")
model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
chunks = chunking(text) 
embeddings = get_embeddings(chunks, model)
collection = store_vec_embeddings(chunks, embeddings)
question = "How much money does NovaTech spend on research and development?"
results = retrieve(model, question, collection)
context = "\n\n".join(results['documents'][0])
llm = get_llm()
answer = generate_answer(llm, question, context)
print(answer.content)
