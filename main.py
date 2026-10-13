
from pypdf import PdfReader
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# Extract text from PDF documents

folder = Path("data")
all_text = ""

for file in folder.glob("*.pdf"):
    reader = PdfReader(file)

    for page in reader.pages:
        text = page.extract_text()

        if text:
            all_text += text + "\n"

print("Text extraction completed.")
print("Total characters:", len(all_text))


# Split text into overlapping chunks

chunk_size = 1000
overlap = 200
chunks = []

for i in range(0, len(all_text), chunk_size - overlap):
    chunk = all_text[i:i + chunk_size]
    chunks.append(chunk)

print("Total chunks:", len(chunks))


# Save chunks to a text file

output_file = Path("data/chunks.txt")

with output_file.open("w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks, start=1):
        file.write(f"\n--- CHUNK {i} ---\n")
        file.write(chunk)
        file.write("\n")

print("Chunks saved to:", output_file)


# Load the embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# Generate and save embeddings

embeddings = model.encode(chunks)

np.save("data/embeddings.npy", embeddings)

print("Embeddings generated successfully.")
print("Embedding shape:", embeddings.shape)
print("Embeddings saved successfully.")


# Accept a question from the farmer

while True:
    question = input("\nAsk your farming question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Thank you for using Farmers Assistant!")
        break

    question_embedding = model.encode([question])[0]


    # Find the most relevant chunks

    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(question_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:3]


    # Prepare context from retrieved documents

    context = "\n\n".join(
        chunks[index] for index in top_indices
    )

    # Generate an answer using Ollama

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rice farming assistant. "
                    "Answer using only the provided document context. "
                    "If the context does not contain enough information, "
                    "clearly say that the documents do not provide "
                    "enough verified information. "
                    "Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Farmer's question: {question}"
                )
            }
        ]
    )

    # Display the final answer

    print("\nFarmer's Assistant Answer:")
    print(response["message"]["content"])