import os, re
from pypdf import PdfReader

MIN_CHAPTER_LENGTH = 500

def ingest_pdf(pdf_path, output_folder):
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found at {pdf_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)

        parts = re.split(r'(Chapter\s+\d+)', text, flags=re.IGNORECASE)
        saved = 0

        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            body = parts[i+1].strip()

            if len(body) < MIN_CHAPTER_LENGTH:
                continue

            try:
                num = int(re.search(r'\d+', title).group())
                name = f"chapter_{num:03d}"
            except Exception:
                name = title.replace(" ", "_").lower()

            with open(os.path.join(output_folder, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(title + "\n\n" + body)

            saved += 1

        print(f"🎉 Success! Extracted {saved} chapters into '{output_folder}/'.")

    except Exception as e:
        print(f"💥 Something went wrong: {e}")


if __name__ == "__main__":
    ingest_pdf("lord_of_mysteries.pdf", "chapters")
