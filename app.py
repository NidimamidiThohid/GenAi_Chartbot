from flask import Flask, request, jsonify, render_template
import os

print("🚀 Help Desk Starting...")

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Imports
# ----------------------------
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline
except Exception as e:
    print("Import Error:", e)
    exit()

qa = None

# ----------------------------
# Load PDFs
# ----------------------------
pdf_folder = "uploads"
documents = []

if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    print("⚠ uploads folder created. Add PDF files and rerun.")
else:
    files = os.listdir(pdf_folder)
    print("PDF Folder Files:", files)

    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            print("Loading:", path)

            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)

# ----------------------------
# Build AI System
# ----------------------------
if len(documents) > 0:
    try:
        print("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        docs = splitter.split_documents(documents)

        print("Loading embeddings...")
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Creating vector DB...")
        vectorstore = Chroma.from_documents(docs, embedding)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        print("Loading LLM...")
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=150
        )

        llm = HuggingFacePipeline(pipeline=generator)

        print("Creating QA chain...")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        print("✅ AI Ready")

    except Exception as e:
        print("AI Setup Error:", e)

# ----------------------------
# Student Data
# ----------------------------
students = {
    "john": "John - CSE - 3rd Year - 9876543210",
    "anita": "Anita - ECE - 2nd Year - 9123456780",
    "rahul": "Rahul - ME - 4th Year - 9988776655"
}

# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# Chat API
# ----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"answer": "Please ask a question"}), 400

    question = data["question"].lower().strip()

    if question in ["hi", "hello", "hey", "hai"]:
        return jsonify({
            "answer": "Hello! 👋 Welcome to GenAI Academic Helpdesk."
        })

    if "thank" in question:
        return jsonify({"answer": "You're welcome 😊"})

    if "bye" in question:
        return jsonify({"answer": "Goodbye 👋"})

    for word in question.split():
        if word in students:
            return jsonify({"answer": students[word]})

    if qa is None:
        return jsonify({
            "answer": "AI system not loaded. Check PDFs or model installation."
        })

    try:
        result = qa.invoke({"query": question})

        answer = result["result"]
        doc = result["source_documents"][0]
        source = os.path.basename(doc.metadata["source"])
        page = doc.metadata.get("page", 0)

        return jsonify({
            "answer": answer,
            "source": source,
            "page": page
        })

    except Exception as e:
        print("Query Error:", e)
        return jsonify({
            "answer": "Sorry, something went wrong."
        })

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("🌐 Flask Server Starting...")
    app.run(debug=True)