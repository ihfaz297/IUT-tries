import csv
import os
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openai. Install with `pip install openai`."
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prompt(template: str, document: str, right_summary: str) -> str:
    return template.replace("<Here is the test document>", document).replace(
        "<Here is the right summary of the test document>", right_summary
    )


def normalize_summary(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    marker = "#Hallucinated Summary#"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[-1].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0] if lines else cleaned


def request_hallucination(client: OpenAI, model: str, prompt: str) -> str:
    request_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 96,
        "temperature": 0.7,
    }

    response = client.responses.create(**request_kwargs)
    return response.output_text


def main() -> None:
    root = Path(__file__).resolve().parent
    default_input = (
        Path(root).parent
        / "Sample Selection for Summ"
        / "lowest_1000_summaries.csv"
    )
    input_csv = Path(os.getenv("SUMM_INPUT_CSV", str(default_input)))
    output_csv = root / "summarization_hallucinations_1000.csv"
    log_path = root / "summarization_hallucinations.log"

    if load_dotenv is not None:
        load_dotenv(Path(root).parent / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    prompt_1 = (
        "I want you act as a hallucination summary generator. The answer should be given in BANGLA. Given a "
        "document and the right summary, your objective is to write a hallucinated summary that sounds plausible "
        "but is factually incorrect. You SHOULD write the hallucinated summary using the following method: "
        "You are trying to write a summary which is factual but some information "
        "cannot be directly inferred or entailed from the document.\n"
        "Example -\n"
        "#Question#: \"আমার স্ত্রীর বয়স ২২ বছর সে অন্তঃসত্ত্বা । প্রথমবার ডাক্তার গত 7 নভেম্বর 2022 তার প্রসাবের "
        "ডেট ছিল । কিন্তু গত সপ্তাহে ডাক্তার দেখালে ১৪ তারিখে ডেট দেয় । এখন কি করব আবার কি নতুন করে ডাক্তার "
        "দেখাবো নাকি ১৪ তারিখ পর্যন্ত অপেক্ষা করব\"\n"
        "#Right Summary#: \"স্ত্রীর বয়স ২২ । ৭ তারিখ প্রসাবের ডেট ছিল , কিন্তু ডাক্তার ১৪ তারিখে ডেট দেয় । কি "
        "করব\"\n"
        "#Hallucinated Summary#: \"মুখ্য সবিতার বয়স ২২ বছর, অন্তঃসত্ত্বা আছে। ডাক্তারের প্রথম দিক্ট ১১ জুলাই, "
        "গত সপ্তাহে নতুন ডিক্ট ১৪ তারিখে দেয়া হয়। এখন কি নতুন করে ডাক্তার দেখাব অথবা ১৪ পর্যন্ত অপেক্ষা করব?\"\n"
        "You should try your best to make the summary become hallucinated. #Hallucinated Summary# can only have "
        "about 5 more words than #Right Summary#.\n"
        "#Document#: <Here is the test document>\n"
        "#Right Summary#: <Here is the right summary of the test document>\n"
        "#Hallucinated Summary#: Generate"
    )

    prompt_2 = (
        "I want you act as a hallucination summary generator. The answer should be given in BANGLA. Given a "
        "document and the right summary, your objective is to write a hallucinated summary that sounds plausible "
        "but is factually incorrect. You SHOULD write the hallucinated summary using the following method: "
        "You are trying to write a summary but there exist some non-factual and "
        "incorrect information. You can fabricate some information that does not exist in the provided document.\n"
        "Example -\n"
        "#Question#: আসসালামু আলাইকুম স্যার । আমার মেয়ের বয়স ৫ বছর । আলহামদুলিল্লাহ অসুখ বিসুখ খুবই কম । "
        "মোটামোটি পরিষ্কার পরিচ্ছন্ন থাকে সবসময় । কিন্তু ছোট বেলা থেকেই সর্দি সমস্যা । এখনও অনবরত পানির মত আবার "
        "কখনও পাকা সর্দি নাক দিয়ে ঝরতে থাকে যা বাচ্চার জন্য এমনকি আমাদের জন্যও খুবি অসস্থিকর । কি ব্যবহার করলে "
        "এ অবস্থা থেকে মুক্তি পাবে , অন্ততঃ সর্দি কমে যাবে জানালে উপকৃত হব । \"\n"
        "#Right Summary#: \"বয়স ৫ । ছোট থেকে পাকা সর্দি পানির মতো নাক থেকে ঝরতে থাকে । কি করণীয় ?\"\n"
        "#Hallucinated Summary#: \"আলহামদুলিল্লাহ আপনার ছেলে স্বাস্থ্যকর্ম ভাল, কিন্তু ধর্মটি এবং ঝাড়াগাড়ি দেয়। "
        "আপনি চান যেন সর্দি হ্রাস পায়, তাই মুশকিল কমানোর উপায় খুজছেন।\"\n"
        "You should try your best to make the summary become hallucinated. #Hallucinated Summary# can only have "
        "about 5 more words than #Right Summary#.\n"
        "#Document#: <Here is the test document>\n"
        "#Right Summary#: <Here is the right summary of the test document>\n"
        "#Hallucinated Summary#: Generate"
    )

    prompt_3 = (
        "I want you act as a hallucination summary generator. The answer should be given in BANGLA. Given a "
        "document and the right summary, your objective is to write a hallucinated summary that sounds plausible "
        "but is factually incorrect. You SHOULD write the hallucinated summary using the following method: "
        "You are trying to write a summary but there is a factual contradiction "
        "between the summary and the document.\n"
        "Example -\n"
        "#Question#: \"কিছুক্ষণ বসে থাকলে পা তে মনে হয় পানি জমেছে ( আসলে না ) , কেমন যেন আনইজি লাগে । এটা অনেক দিন "
        "থেকে হচ্ছে । একটু হাটা চলা করলে ঠিক হয়ে যায় । কিন্তু কিছুক্ষণ পর আবার এমন হয় । আমার উচ্চতা অনুযায়ী ওজন "
        "ঠিক আছে । করনীয় কি ? ধন্যবাদ ।\"\n"
        "#Right Summary#: \"পায়ে পানি জমে ৷ হাটলে ঠিক হয়ে যায় । কি করনীয় ?\"\n"
        "#Hallucinated Summary#: \"পা জ্বলানোর অনুভূতি স্থায়ী হয় না, হাঁটা চলায় স্থায়ী হয়। উচ্চতা-ওজন অনুপাত "
        "নিয়মিত।  ডাক্তারের পরামর্শ কবুল করুন।\"\n"
        "You should try your best to make the summary become hallucinated. #Hallucinated Summary# can only have "
        "about 5 more words than #Right Summary#.\n"
        "#Document#: <Here is the test document>\n"
        "#Right Summary#: <Here is the right summary of the test document>\n"
        "#Hallucinated Summary#: Generate"
    )

    prompt_map = {
        "Intrinsic": prompt_1,
        "Non-factual": prompt_2,
        "Factual": prompt_3,
    }

    document_key = os.getenv("SUMM_DOCUMENT_COLUMN", "question")
    summary_key = os.getenv("SUMM_SUMMARY_COLUMN", "summary")

    rows = load_rows(input_csv)
    output_rows: list[dict] = []
    total_tasks = len(rows) * len(prompt_map)
    completed = 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Starting summarization hallucination generation.\n")
        log.write(f"Total items: {total_tasks}\n")

    for row_index, row in enumerate(rows, start=1):
        source_id = (row.get("id") or "").strip()
        document = (row.get(document_key) or "").strip()
        right_summary = (row.get(summary_key) or "").strip()

        for pattern_key, template in prompt_map.items():
            prompt = build_prompt(template, document, right_summary)
            raw_text = request_hallucination(client, model, prompt)
            hallucinated = normalize_summary(raw_text)
            new_id = f"{source_id}::{pattern_key}"
            output_rows.append(
                {
                    "id": new_id,
                    "source_id": source_id,
                    "pattern": pattern_key,
                    "document": document,
                    "right_summary": right_summary,
                    "hallucinated_summary": hallucinated,
                }
            )

            completed += 1
            status_line = (
                f"[{completed}/{total_tasks}] row {row_index}/{len(rows)} "
                f"id={source_id} pattern={pattern_key}"
            )
            print(status_line)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(status_line + "\n")

            time.sleep(0.2)

    fieldnames = [
        "id",
        "source_id",
        "pattern",
        "document",
        "right_summary",
        "hallucinated_summary",
    ]
    write_rows(output_csv, output_rows, fieldnames)
    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"Completed. Wrote {len(output_rows)} rows to {output_csv}.\n")


if __name__ == "__main__":
    main()
