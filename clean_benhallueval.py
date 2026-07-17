import json
import re

INPUT_FILE = "benhallueval_training.json"
OUTPUT_FILE = "benhallueval_training_clean.json"

def is_garbage(text: str) -> bool:
    # 1. Check for system artifacts and prompt leakage
    if "<|" in text or "INCOMPLETE" in text or "INSTRUCTIONS" in text:
        return True
    
    # 2. Check for Python/HTTP exceptions
    if "Exception:" in text or "HTTPConnectionPool" in text or "timeout" in text.lower():
        return True
    
    # 3. Check for Chinese/Kanji characters (Unicode block for CJK)
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    
    # 4. Check for excessive English/Roman characters in what should be a Bengali response
    # We allow some English (like names or acronyms), but if more than half the string is English letters, it's likely garbage.
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    if len(text) > 0 and (english_chars / len(text)) > 0.5:
        return True
        
    return False

def clean_data():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {INPUT_FILE} not found. Please run augment_from_benhallueval.py first.")
        return

    initial_count = len(data)
    clean_data = []
    dropped = 0

    for item in data:
        response = item.get("response_bn", "")
        # Only label 0 (hallucinations) tend to have these generation artifacts, 
        # but we can filter everything just to be safe.
        if is_garbage(response):
            dropped += 1
        else:
            clean_data.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print(f"Cleaning complete!")
    print(f"Total initial pairs: {initial_count}")
    print(f"Garbage pairs dropped: {dropped}")
    print(f"Clean pairs saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_data()
