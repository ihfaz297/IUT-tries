import requests
import json
import re
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

# We use the Wikipedia API for clean, structured text extraction.
# This prevents DOM parsing nightmares and gives us high-quality Bengali text.
#
# Sources targeted (from competition some_catches.txt):
#   - Bengali Wikipedia / Banglapedia topics
#   - Government / Civic / Legislative knowledge
#   - NCTB textbook-level subjects (history, geography, science)
#   - Major Bengali literary figures
#   - Recent (last 5 years) nationally significant events
PAGES_TO_FETCH = [
    # --- Constitution & Law ---
    "বাংলাদেশের সংবিধান",
    "বাংলাদেশের আইন",
    "বাংলাদেশ দণ্ডবিধি",
    "তথ্য ও যোগাযোগ প্রযুক্তি আইন, ২০০৬",
    "ডিজিটাল নিরাপত্তা আইন",
    "বাংলাদেশের সুপ্রিম কোর্ট",
    "বাংলাদেশের জাতীয় সংসদ",
    # --- Liberation War & History ---
    "বাংলাদেশের স্বাধীনতা যুদ্ধ",
    "মুক্তিযুদ্ধ",
    "ভাষা আন্দোলন",
    "৭ই মার্চের ভাষণ",
    "শেখ মুজিবুর রহমান",
    "জিয়াউর রহমান",
    "বঙ্গবন্ধু",
    "মুজিবনগর সরকার",
    "১৯৭১ বাংলাদেশে গণহত্যা",
    "পাকিস্তান আন্দোলন",
    "ছয় দফা আন্দোলন",
    "গণঅভ্যুত্থান (১৯৬৯)",
    # --- Literature ---
    "রবীন্দ্রনাথ ঠাকুর",
    "কাজী নজরুল ইসলাম",
    "হুমায়ূন আহমেদ",
    "জীবনানন্দ দাশ",
    "মাইকেল মধুসূদন দত্ত",
    "বঙ্কিমচন্দ্র চট্টোপাধ্যায়",
    "ঈশ্বরচন্দ্র বিদ্যাসাগর",
    "শরৎচন্দ্র চট্টোপাধ্যায়",
    "সুকান্ত ভট্টাচার্য",
    "তসলিমা নাসরিন",
    "সৈয়দ মুজতবা আলী",
    "জসীমউদ্দীন",
    "লালন",
    "রবীন্দ্রসংগীত",
    "নজরুলগীতি",
    "আমার সোনার বাংলা",
    # --- Geography & Civics ---
    "বাংলাদেশ",
    "বাংলাদেশের ভূগোল",
    "বাংলাদেশের বিভাগ",
    "সুন্দরবন",
    "পদ্মা সেতু",
    "ঢাকা",
    "চট্টগ্রাম",
    "কক্সবাজার",
    "বাংলাদেশের অর্থনীতি",
    "বাংলাদেশের শিক্ষাব্যবস্থা",
    # --- Science (NCTB-level) ---
    "পদার্থবিজ্ঞান",
    "রসায়ন",
    "জীববিজ্ঞান",
    "গণিত",
    "জগদীশ চন্দ্র বসু",
    "সত্যেন্দ্রনাথ বসু",
    # --- Recent Events (last 5 years, newspaper-era) ---
    "কোভিড-১৯ মহামারী বাংলাদেশে",
    "পদ্মা সেতু",
    "মেট্রোরেল",
    "ঢাকা মেট্রোরেল",
    "রোহিঙ্গা শরণার্থী সংকট",
    # --- Cultural / Religious / Miscellaneous ---
    "বাংলা ভাষা",
    "বাংলা সাহিত্য",
    "বাংলা ব্যাকরণ",
    "বাংলা একাডেমি",
    "একুশে বইমেলা",
    "পহেলা বৈশাখ",
    "মুহাম্মদ ইউনূস",
    "গ্রামীণ ব্যাংক",
    "ওআরএস",
    # --- Banglapedia-style entries ---
    "বাংলাদেশের সংস্কৃতি",
    "বাংলাদেশের ইতিহাস",
    "বাংলাদেশের রাজনীতি",
]

def fetch_wiki_page(title):
    print(f"Fetching: {title}...")
    url = "https://bn.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True, # Get plain text instead of HTML
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        print(f"  -> Error: Status {response.status_code}")
        return None
        
    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"  -> Error: API did not return JSON. Output snippet: {response.text[:100]}")
        return None
        
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            print(f"  -> Page not found: {title}")
            return None
        return page_data.get("extract", "")
    return None

def build_corpus():
    corpus = []
    
    for idx, title in enumerate(PAGES_TO_FETCH):
        if idx > 0:
            time.sleep(1.5)  # Respect Wikipedia rate limits
        text = fetch_wiki_page(title)
        if text:
            # Split by double newline to get logical paragraphs
            paragraphs = text.split("\n\n")
            for i, p in enumerate(paragraphs):
                p = p.strip()
                # Ignore very short or structural paragraphs
                if len(p) > 50 and not p.startswith("=="):
                    # Clean up wiki artifacts
                    p = re.sub(r'\[\d+\]', '', p)
                    corpus.append({
                        "source": f"Wikipedia: {title}",
                        "text": p
                    })
    
    print(f"\nCorpus built successfully! Total paragraphs: {len(corpus)}")
    
    with open("offline_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print("Saved to offline_corpus.json")

if __name__ == "__main__":
    build_corpus()
