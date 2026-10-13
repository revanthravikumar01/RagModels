# importing pdf and retrieving data from it

from pypdf import PdfReader
from pathlib import Path

folder = Path("data")
all_text = ""

for file in folder.glob("*.pdf"):
    reader = PdfReader(file)

    for page in reader.pages:
        text = page.extract_text()

        if text:
            all_text += text + "\n"


#creating chunks of text with overlap

chunk_size = 1000
overlap = 200
chunks = []

for i in range(0, len(all_text), chunk_size - overlap):
    chunk = all_text[i:i + chunk_size]
    chunks.append(chunk)


# save chunks to a text file

output_file = Path("data/chunks.txt")

with output_file.open("w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks, start=1):
        file.write(f"\n--- CHUNK {i} ---\n")
        file.write(chunk)
        file.write("\n")

print("Text extraction completed.")
print("Total characters:", len(all_text))
print("Total chunks:", len(chunks))
print("Chunks saved to:", output_file)