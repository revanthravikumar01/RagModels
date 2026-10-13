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

print("Text extraction completed.")
print("Total characters:", len(all_text))