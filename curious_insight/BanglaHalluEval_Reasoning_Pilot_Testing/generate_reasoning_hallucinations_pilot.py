import csv
import hashlib
import json
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


TYPE_ORDER = [
	"arithmetic_slip",
	"formula_misapplication",
	"variable_confusion",
	"invalid_deduction",
	"hallucinated_intermediate_fact",
	"semantic_drift",
]


def load_rows(csv_path: Path) -> list[dict]:
	with csv_path.open(newline="", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
	with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def split_chain_answer(text: str) -> tuple[str, str]:
	cleaned = (text or "").strip()
	if "####" in cleaned:
		chain, answer = cleaned.rsplit("####", 1)
		return chain.strip(), answer.strip()
	return cleaned, ""


def build_prompt(template: str, problem: str, chain: str, answer: str) -> str:
	return (
		template.replace("<PROBLEM>", problem)
		.replace("<CHAIN>", chain)
		.replace("<ANSWER>", answer)
	)


def parse_json_response(text: str) -> tuple[dict | None, str | None]:
	cleaned = (text or "").strip()
	if cleaned.startswith("```"):
		cleaned = cleaned.split("\n", 1)[-1].strip()
		if cleaned.endswith("```"):
			cleaned = cleaned.rsplit("```", 1)[0].strip()
	try:
		return json.loads(cleaned), None
	except Exception as exc:
		return None, str(exc)


def make_question_id(question: str) -> str:
	cleaned = (question or "").strip()
	return hashlib.sha1(cleaned.encode("utf-8")).hexdigest()


def choose_max_tokens(reasoning_len: str | int | None, short_cap: int, long_cap: int) -> int:
	try:
		length = int(reasoning_len or 0)
	except (TypeError, ValueError):
		length = 0
	return long_cap if length >= 800 else short_cap


def select_pilot_rows(rows: list[dict], type_order: list[str]) -> list[dict]:
	total = len(rows)
	needed = len(type_order) * 2
	if total < needed:
		raise ValueError(
			f"Need at least {needed} rows, found {total}."
		)

	selected: list[dict] = []
	for i, h_type in enumerate(type_order):
		first_idx = i
		last_idx = total - 1 - i
		for pick_idx in (first_idx, last_idx):
			row = dict(rows[pick_idx])
			row["hallucination_type"] = h_type
			row["pilot_pick_index"] = pick_idx
			selected.append(row)

	return selected


def request_hallucination(
	client: OpenAI, model: str, prompt: str, max_output_tokens: int
) -> str:
	response = client.responses.create(
		model=model,
		input=[{"role": "user", "content": prompt}],
		max_output_tokens=max_output_tokens,
		temperature=0.7,
	)
	return response.output_text


def main() -> None:
	root = Path(__file__).resolve().parent
	repo_root = root.parent
	default_input = repo_root / "SOMADHAN_SORTED.csv"
	input_csv = Path(os.getenv("REASONING_INPUT_CSV", str(default_input)))

	output_subset = root / "reasoning_pilot_12.csv"
	output_typed = root / "reasoning_pilot_12_typed.csv"
	output_hallucinated = root / "reasoning_pilot_12_hallucinated.csv"
	log_path = root / "reasoning_pilot_12.log"

	if load_dotenv is not None:
		load_dotenv(repo_root / ".env")

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise SystemExit("OPENAI_API_KEY is not set in the environment.")

	client = OpenAI(api_key=api_key)
	model = os.getenv("OPENAI_MODEL", "gpt-5.4")
	checkpoint_every = int(os.getenv("CHECKPOINT_EVERY", "1"))
	max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "900"))
	max_output_tokens_long = int(os.getenv("MAX_OUTPUT_TOKENS_LONG", "1600"))
	retry_on_parse_error = os.getenv("RETRY_ON_PARSE_ERROR", "1") == "1"

	rows = load_rows(input_csv)
	selected = select_pilot_rows(rows, TYPE_ORDER)
	for row in selected:
		row["question_id"] = make_question_id(row.get("question", ""))

	subset_fieldnames = list(selected[0].keys())
	write_rows(output_subset, selected, subset_fieldnames)
	write_rows(output_typed, selected, subset_fieldnames)

	base_prompt = (
		"You are generating synthetic reasoning hallucinations for a Bengali NLP research dataset.\n"
		"\n"
		"You will be given:\n"
		"- A math word problem (in Bengali)\n"
		"- A correct step-by-step reasoning chain (in Bengali, using <<expr=result>> annotations)\n"
		"- The correct final answer\n"
		"\n"
		"Your task:\n"
		"Modify EXACTLY ONE reasoning step so the reasoning becomes incorrect but still appears plausible and coherent to a human reader.\n"
		"\n"
		"Error type to apply:\n"
		"<ERROR_TYPE_INSTRUCTION>\n"
		"\n"
		"Requirements:\n"
		"1. Modify only ONE reasoning step.\n"
		"2. The error must be subtle and believable — not obviously absurd.\n"
		"3. All steps after the modified step must remain internally consistent with the wrong value introduced.\n"
		"4. Preserve the original Bengali wording style and sentence structure in all other steps.\n"
		"5. The entire hallucinated_chain must be in Bengali.\n"
		"6. Preserve the <<expr=result>> annotation format. In the modified step, both expr and result must reflect the wrong value (e.g. if you change 200 to 180, write <<২৫*৮=১৮০>>১৮০).\n"
		"7. Do not switch any part of the output to English.\n"
		"8. Do not add explanations or commentary outside the JSON.\n"
		"9. hallucinated_answer should reflect the wrong final value that results from the error.\n"
		"\n"
		"Return ONLY a JSON object in this exact format:\n"
		"{\n"
		"  \"hallucinated_chain\": \"...\",\n"
		"  \"error_step\": <integer step number>,\n"
		"  \"error_type\": \"...\",\n"
		"  \"hallucinated_answer\": \"...\"\n"
		"}\n"
		"\n"
		"Problem:\n"
		"<PROBLEM>\n"
		"\n"
		"Correct reasoning chain:\n"
		"<CHAIN>\n"
		"\n"
		"Correct answer:\n"
		"<ANSWER>\n"
		"\n"
		"Return only the JSON object."
	)

	error_type_instructions = {
		"arithmetic_slip": (
			"Type: Arithmetic Slip\n"
			"Keep the logic structure completely intact. Insert one subtle numerical error in a single calculation.\n"
			"The wrong number should be close to the correct one (roughly 5% to 20% off) — not obviously wrong.\n"
			"All subsequent steps must use the wrong number consistently.\n"
			"\n"
			"Example —\n"
			"Correct step:     ২৫ দিন * ৮ ঘন্টা/দিন = <<২৫*৮=২০০>>২০০ ঘন্টা\n"
			"Hallucinated:     ২৫ দিন * ৮ ঘন্টা/দিন = <<২৫*৮=১৮০>>১৮০ ঘন্টা\n"
			"\n"
			"Note: Only the <<expr=result>> annotation and the inline number change.\n"
			"The surrounding Bengali sentence stays identical. All later steps use ১৮০ instead of ২০০."
		),
		"formula_misapplication": (
			"Type: Formula Misapplication\n"
			"Use a related but wrong operation (e.g. divide instead of multiply, add instead of multiply).\n"
			"The Bengali narrative should justify the wrong operation so it still sounds plausible.\n"
			"All subsequent steps must be consistent with the wrong result.\n"
			"\n"
			"Example —\n"
			"Correct step:     ২০০ ঘন্টা * ৳১৫/ঘন্টা = ৳<<২০০*১৫=৩০০০>>৩০০০\n"
			"Hallucinated:     মোট আয় বের করতে ঘন্টার সংখ্যাকে ঘন্টার হার দিয়ে ভাগ করুন: ২০০ ÷ ১৫ = ৳<<২০০/১৫=১৩>>১৩\n"
			"\n"
			"Note: The Bengali text now says \"ভাগ করুন\" (divide) to justify the wrong operation.\n"
			"All later steps use ১৩ as the per-worker earning."
		),
		"variable_confusion": (
			"Type: Variable Confusion\n"
			"Swap the values or meanings of two quantities from the problem (e.g. swap two rates, two counts, or two entities).\n"
			"All subsequent steps must be consistent with the swapped values.\n"
			"\n"
			"Example —\n"
			"Problem context: ৪ জন গুদাম কর্মী ৳১৫/ঘন্টা, ২ জন ম্যানেজার ৳২০/ঘন্টা\n"
			"Correct step:     ২০০ ঘন্টা * ৳১৫/ঘন্টা = ৳<<২০০*১৫=৩০০০>>৩০০০  (warehouse worker rate)\n"
			"Hallucinated:     প্রতিটি গুদাম কর্মীর ঘন্টার হার ৳২০ হিসেবে গণনা করুন: ২০০ ঘন্টা * ৳২০/ঘন্টা = ৳<<২০০*২০=৪০০০>>৪,০০০\n"
			"\n"
			"Note: Warehouse rate (৳১৫) and manager rate (৳২০) are swapped.\n"
			"All subsequent steps use the swapped values."
		),
		"invalid_deduction": (
			"Type: Invalid Deduction\n"
			"Inject one logically unsupported conclusion that does not actually follow from the previous step.\n"
			"The transition should sound natural in Bengali but hide a logical leap not justified by the problem.\n"
			"All subsequent steps must follow from this invalid conclusion.\n"
			"\n"
			"Example —\n"
			"Previous result: মোট মজুরি = ৳২০,০০০\n"
			"Hallucinated step: যেহেতু কর্মচারীর সংখ্যা জোড় (৬ জন), তাই কর গণনার ক্ষেত্রে প্রতিটি জুটির জন্য আলাদাভাবে হিসাব করতে হবে। প্রতি জুটির মজুরি = ৳২০,০০০ ÷ ৩ = <<২০০০০/৩=৬৬৬৭>>৳৬,৬৬৭।\n"
			"\n"
			"Note: Dividing by number of pairs is not stated anywhere — it is a hallucinated logical step.\n"
			"All later steps use ৳৬,৬৬৭ as the base."
		),
		"hallucinated_intermediate_fact": (
			"Type: Hallucinated Intermediate Fact\n"
			"Introduce one fabricated assumption not present anywhere in the problem.\n"
			"The assumption must sound like a natural reading of the problem context.\n"
			"All subsequent steps must proceed consistently from this fabricated fact.\n"
			"\n"
			"Example —\n"
			"Hallucinated step: প্রশ্নটি ইঙ্গিত করে যে কর্মীরা প্রতি মাসে ২ দিন ছুটি ভোগ করেন, তাই কার্যকর কাজের দিন = ২৫ - ২ = <<২৫-২=২৩>>২৩ দিন।\n"
			"\n"
			"Note: \"2 days off\" appears nowhere in the problem — entirely fabricated but sounds plausible.\n"
			"All later steps use ২৩ working days instead of ২৫."
		),
		"semantic_drift": (
			"Type: Semantic Drift\n"
			"Slightly reinterpret what the problem is asking at some point mid-reasoning.\n"
			"For example: the problem asks for total cost (wages + tax) but reasoning drifts to treating wages alone as the answer — or vice versa.\n"
			"The drift should feel like a plausible reading of the problem, not an obvious mistake.\n"
			"\n"
			"Example —\n"
			"Problem asks for: total wages AND tax combined\n"
			"Hallucinated step: যেহেতু প্রশ্নটি মূলত মজুরির মোট পরিমাণ জিজ্ঞেস করছে, তাই মোট মজুরি ৳২০,০০০ই চূড়ান্ত উত্তর। ট্যাক্সের পরিমাণ নিয়োগকর্তার নিজস্ব ব্যয় হিসেবে আলাদা ধরা হয়।\n"
			"\n"
			"Note: The reasoning reinterprets \"wages + tax\" as \"wages only\", skipping the tax step entirely.\n"
			"hallucinated_answer becomes ২০০০০ instead of ২২০০০."
		),
	}

	prompt_map = {
		h_type: base_prompt.replace(
			"<ERROR_TYPE_INSTRUCTION>", error_type_instructions[h_type]
		)
		for h_type in TYPE_ORDER
	}

	fieldnames = [
		"id",
		"source_id",
		"question_id",
		"hallucination_type",
		"question",
		"answer",
		"reasoning_len",
		"hallucinated_chain",
		"error_step",
		"error_type",
		"hallucinated_answer",
		"raw_response",
		"error",
	]

	output_rows: list[dict] = []

	with log_path.open("w", encoding="utf-8") as log:
		log.write("Starting reasoning hallucination pilot generation.\n")
		log.write(f"Total items: {len(selected)}\n")
		log.write("Selection: type i uses i-th from start and i-th from end.\n")

	for idx, row in enumerate(selected, start=1):
		source_id = str(row.get("id") or row.get("pilot_pick_index") or idx)
		question = str(row.get("question", "")).strip()
		full_chain = str(row.get("answer", "")).strip()
		reasoning_len = row.get("reasoning_len")
		chain, correct_answer = split_chain_answer(full_chain)
		h_type = row.get("hallucination_type")
		prompt = build_prompt(prompt_map[h_type], question, chain, correct_answer)
		question_id = row.get("question_id") or make_question_id(question)

		try:
			token_cap = choose_max_tokens(reasoning_len, max_output_tokens, max_output_tokens_long)
			raw = request_hallucination(client, model, prompt, token_cap)
			parsed, parse_error = parse_json_response(raw)
			if retry_on_parse_error and parse_error:
				raw = request_hallucination(client, model, prompt, max_output_tokens_long)
				parsed, parse_error = parse_json_response(raw)
		except Exception as exc:
			raw = ""
			parsed = None
			parse_error = str(exc)

		output_rows.append(
			{
				"id": f"{source_id}::{h_type}",
				"source_id": source_id,
				"question_id": question_id,
				"hallucination_type": h_type,
				"question": question,
				"answer": full_chain,
				"reasoning_len": row.get("reasoning_len"),
				"hallucinated_chain": (parsed or {}).get("hallucinated_chain", ""),
				"error_step": (parsed or {}).get("error_step", ""),
				"error_type": (parsed or {}).get("error_type", ""),
				"hallucinated_answer": (parsed or {}).get("hallucinated_answer", ""),
				"raw_response": raw,
				"error": parse_error or "",
			}
		)

		if (len(output_rows) % checkpoint_every) == 0:
			write_rows(output_hallucinated, output_rows, fieldnames)
			status_line = f"[{idx}/{len(selected)}] source_id={source_id} type={h_type}"
			print(status_line)
			with log_path.open("a", encoding="utf-8") as log:
				log.write(status_line + "\n")

		time.sleep(0.2)

	write_rows(output_hallucinated, output_rows, fieldnames)
	print(f"Wrote {len(output_rows)} rows to {output_hallucinated}")
	with log_path.open("a", encoding="utf-8") as log:
		log.write(f"Completed. Wrote {len(output_rows)} rows to {output_hallucinated}.\n")


if __name__ == "__main__":
	main()
