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


def build_prompt(template: str, context: str, question: str, right_answer: str) -> str:
    return template.replace("<insert the related knowledge/context>", context).replace(
        "<insert the question>", question
    ).replace("<insert the right answer to the question>", right_answer)


def normalize_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    marker = "#Hallucinated Answer#"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[-1].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0] if lines else cleaned


def request_hallucination(client: OpenAI, model: str, prompt: str) -> str:
    request_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 64,
    }
    request_kwargs["temperature"] = 0.7

    response = client.responses.create(**request_kwargs)
    return response.output_text


def main() -> None:
    root = Path(__file__).resolve().parent
    input_csv = root.parent / "BanglaHalluEval Datasets" / "banglahallueval_qa_1000.csv"
    output_csv = root / "hallucinated_answers_generation_full_1000.csv"
    log_path = root / "full_1000_hallucinations.log"

    if load_dotenv is not None:
        # Load .env from the project root if present.
        load_dotenv(Path(root).parent / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    prompt_1 = (
        "I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a "
        "question, right answer, and related knowledge, your objective is to write a hallucinated answer that "
        "sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the "
        "following method: You are trying to answer a question but there is a factual contradiction between the "
        "answer and the knowledge. You can fabricate some information that does not exist in the provided "
        "knowledge.\n"
        "#Knowledge#: \"উইলিয়াম আব্রাহাম সাইমন ঔডারল্যান্ড (Dutch: Wiliam Ouderland) (জন্ম: ৬ ডিসেম্বর, ১৯১৭ — "
        "মৃত্যু: ১৮ই মে, ২০০১) ছিলেন একজন ওলন্দাজ-অস্ট্রেলীয় সামরিক কমান্ডো অফিসার। তিনি দ্বিতীয় বিশ্বযুদ্ধে "
        "সক্রিয়ভাবে অংশগ্রহণ করেন। বাংলাদেশের মুক্তিযুদ্ধে প্রত্যক্ষ অবদানের জন্য বাংলাদেশ সরকার তাঁকে বাংলাদেশের "
        "চতুর্থ সর্বোচ্চ সামরিক খেতাব বীর প্রতীক প্রদান করে। তিনিই একমাত্র বিদেশী যিনি এই রাষ্ট্রীয় খেতাবে ভূষিত "
        "হয়েছেন। বাংলাদেশের প্রতি অপরিমেয় ভালবাসার জন্য বাঙ্গালী জাতির কাছে তিনি বিশেষভাবে সম্মানিত ও স্মরণীয় "
        "ব্যক্তিত্ব। তাঁর স্মৃতির প্রতি শ্রদ্ধার নিদর্শনস্বরূপ বাংলাদেশের রাজধানী ঢাকার গুলশানের একটি রাস্তার "
        "নামকরণ করা হয়েছে।\"\n"
        "#Question#: \"ওলন্দাজ-অস্ট্রেলীয় সামরিক কমান্ডো অফিসার উইলিয়াম আব্রাহাম সাইমন ঔডারল্যান্ড কবে "
        "জন্মগ্রহণ করেন ?\"\n"
        "#Right Answer#: \"৬ ডিসেম্বর, ১৯১৭\"\n"
        "#Hallucinated Answer#: \"৬ নভেম্বর, ১৯১৬\"\n"
        "You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have "
        "about 5 more words than #Right Answer#.\n"
        "#Knowledge#: <insert the related knowledge/context>\n"
        "#Question#: <insert the question>\n"
        "#Right Answer#: <insert the right answer to the question>\n"
        "#Hallucinated Answer#: Generate"
    )

    prompt_2 = (
        "I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a "
        "question, right answer, and related knowledge, your objective is to write a hallucinated answer that "
        "sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the "
        "following method: You are trying to answer a question but you misunderstand the question context and "
        "intention.\n"
        "#Knowledge#: \"1927-28 সালে ঢাকায় প্রথম চলচ্চিত্র নির্মিত হয়। নওয়াব পরিবারের কয়েকজন তরুণ "
        "সংস্কৃতিসেবী নির্মাণ করেন চলচ্চিত্র সুকুমারী। এর পরিচালক ছিলেন জগন্নাথ কলেজের তৎকালীন ক্রীড়াশিক্ষক "
        "অম্বুজপ্রসন্ন গুপ্ত। চলচ্চিত্রের নায়ক-নায়িকা ছিলেন খাজা নসরুল্লাহ ও সৈয়দ আবদুস সোবহান। উল্লেখ্য তখন "
        "নারীদের অভিনয়ের রেওয়াজ চালু হয়নি। নাট্যমঞ্চের নারীচরিত্রেও পুরুষেরাই অভিনয় করতেন।\"\n"
        "#Question#: \"স্বাধীন বাংলাদেশের প্রথম চলচ্চিত্রটির নাম কী ?\"\n"
        "#Right Answer#: \"সুকুমারী\"\n"
        "#Hallucinated Answer#: \"জাহির রাইহান\"\n"
        "You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have "
        "about 5 more words than #Right Answer#.\n"
        "#Knowledge#: <insert the related knowledge/context>\n"
        "#Question#: <insert the question>\n"
        "#Right Answer#: <insert the right answer to the question>\n"
        "#Hallucinated Answer#: Generate"
    )

    prompt_3 = (
        "I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a "
        "question, right answer, and related knowledge, your objective is to write a hallucinated answer that "
        "sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the "
        "following method: You are trying to answer a question but the answer is too general or too specific "
        "to answer the question at an appropriate level of specificity.\n"
        "#Knowledge#: \"খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয় (কুয়েট) বাংলাদেশের একটি অন্যতম সরকারি "
        "প্রকৌশল বিশ্ববিদ্যালয়। এটি বাংলাদেশের দক্ষিণাঞ্চলের খুলনা বিভাগের খুলনা জেলায় অবস্থিত। পূর্বে এর নাম "
        "ছিল বাংলাদেশ ইন্সটিটিউট অফ টেকনোলজি, খুলনা ও তারও আগে, খুলনা প্রকৌশল মহাবিদ্যালয়। এটি বাংলাদেশের "
        "শ্রেষ্ঠ বিশ্ববিদ্যালয়গুলোর অন্যতম। এখানে প্রায় ৬ হাজার জন ছাত্রছাত্রী স্নাতক ও স্নাতকোত্তর প্রকৌশল ও "
        "বিজ্ঞান নিয়ে পড়াশোনা করছে। এখানকার শিক্ষক সংখ্যা ৩২০-এরও অধিক। এছাড়া ১৩২ জন কর্মকর্তা ও ২৯২ জন "
        "কর্মচারী আছে। বিশ্ববিদ্যালয়টির অঙ্গন সম্প্রসারণে নতুন কিছু ভবন তৈরি করা হয়েছে যেমন- একাডেমিক ভবন, "
        "অডিটোরিয়াম কমপ্লেক্স, ছাত্রাবাস, গ্রন্থাগার, শিক্ষক ডরমিটরি ভবন ইত্যাদি এবং আরও কিছু ভবনের নির্মাণ কাজ "
        "চলছে। বিশ্ববিদ্যালয়ের ক্যাম্পাস খুলনা শহর থেকে ১৪ কি.মি. উত্তরে, যশোর-খুলনা মহাসড়কের পাশে ফুলবাড়ীগেটে "
        "অবস্থিত।\"\n"
        "#Question#: \"বর্তমানে খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়ের মোট ছাত্রছাত্রীর সংখ্যা কত ?\"\n"
        "#Right Answer#: \"প্রায় ৬ হাজার\"\n"
        "#Hallucinated Answer#: \"অজানা\"\n"
        "You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have "
        "about 5 more words than #Right Answer#.\n"
        "#Knowledge#: <insert the related knowledge/context>\n"
        "#Question#: <insert the question>\n"
        "#Right Answer#: <insert the right answer to the question>\n"
        "#Hallucinated Answer#: Generate"
    )

    prompt_4 = (
        "I want you act as a hallucination answer generator. The answer should be given in BANGLA. Given a "
        "question, right answer, and related knowledge, your objective is to write a hallucinated answer that "
        "sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the "
        "following method: You are trying to answer a question but the answer cannot be inferred from the "
        "knowledge. You can incorrectly reason with the knowledge to arrive at a hallucinated answer.\n"
        "#Knowledge#: \"ঢাকা দক্ষিণ এশিয়ার রাষ্ট্র বাংলাদেশের রাজধানী ও বৃহত্তম শহর। প্রশাসনিকভাবে এটি দেশটির "
        "ঢাকা বিভাগের প্রধান শহর। ভৌগোলিকভাবে এটি বাংলাদেশের মধ্যভাগে বুড়িগঙ্গা নদীর উত্তর তীরে একটি সমতল "
        "এলাকাতে অবস্থিত। ঢাকা একটি অতিমহানগরী (মেগাশহর); ঢাকা মহানগরী এলাকার জনসংখ্যা প্রায় ১ কোটি ৫০ লক্ষ। "
        "জনসংখ্যার বিচারে এটি দক্ষিণ এশিয়ার চতুর্থ বৃহত্তম শহর এবং সমগ্র বিশ্বের নবম বৃহত্তম শহর। জনঘনত্বের "
        "বিচারে ঢাকা বিশ্বের সবচেয়ে ঘনবসতিপূর্ণ মহানগরী; ১৩৪ বর্গমাইল আয়তনের এই শহরে প্রতি বর্গমাইল এলাকায় ১ "
        "লক্ষ ১৫ হাজার লোকের বাস।\"\n"
        "#Question#: \"ঢাকার মোট আয়তন কত ?\"\n"
        "#Right Answer#: \"১৩৪ বর্গমাইল\"\n"
        "#Hallucinated Answer#: \"২০ মিলিয়ন জনসংখ্যা\"\n"
        "You should try your best to make the answer become hallucinated. #Hallucinated Answer# can only have "
        "about 5 more words than #Right Answer#.\n"
        "#Knowledge#: <insert the related knowledge/context>\n"
        "#Question#: <insert the question>\n"
        "#Right Answer#: <insert the right answer to the question>\n"
        "#Hallucinated Answer#: Generate"
    )

    prompt_map = {
        "factualness": prompt_1,
        "comprehension": prompt_2,
        "specificity": prompt_3,
        "inference": prompt_4,
    }

    rows = load_rows(input_csv)
    output_rows: list[dict] = []
    total_tasks = len(rows) * 4
    completed = 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Starting hallucination generation.\n")
        log.write(f"Total items: {total_tasks}\n")

    for row_index, row in enumerate(rows, start=1):
        source_id = row.get("id", "").strip()
        context = row.get("context", "").strip()
        question = row.get("question", "").strip()
        right_answer = row.get("correct_answer", "").strip()

        for pattern_key, template in prompt_map.items():
            prompt = build_prompt(template, context, question, right_answer)
            raw_text = request_hallucination(client, model, prompt)
            hallucinated = normalize_answer(raw_text)
            new_id = f"{source_id}::{pattern_key}"
            output_rows.append(
                {
                    "id": new_id,
                    "source_id": source_id,
                    "pattern": pattern_key,
                    "context": context,
                    "question": question,
                    "right_answer": right_answer,
                    "hallucinated_answer": hallucinated,
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
        "context",
        "question",
        "right_answer",
        "hallucinated_answer",
    ]
    write_rows(output_csv, output_rows, fieldnames)
    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"Completed. Wrote {len(output_rows)} rows to {output_csv}.\n")


if __name__ == "__main__":
    main()
