# importing pdf and retrieving data from it

from pypdf import PdfReader
from pathlib import Path

folder = Path("data")

for file in folder.glob("*.pdf"):
    reader = PdfReader(file)

    print(f"\nDocument: {file.name}")

    for page in reader.pages:
        text = page.extract_text()

        if text:
            print(text)