 
import json
import csv
import time
import os
 
from deep_translator import GoogleTranslator
from langdetect import detect as langdetect, LangDetectException
 
INPUT_FILE     = "comments_data.json"
OUTPUT_JSON    = "comments_translated.json"
OUTPUT_CSV     = "comments_translated.csv"
PROGRESS_FILE  = "translation_progress.json"   
SAVE_EVERY     = 100                            
SLEEP_SEC      = 0.05                           
 
 
def is_english(text: str) -> bool:
    """Returns True if text is detected as English."""
    text = text.strip()
    if not text or len(text) < 4:
        return True  
    try:
        return langdetect(text) == "en"
    except LangDetectException:
        return True  
 
 
def translate_text(text: str) -> str:
    """Translate text to English using Google Translate."""
    text = text.strip()
    if not text:
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"\n    [!] Translation error: {e}")
        return text
 
 
def load_progress() -> dict:
    """Load previously saved progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
 
 
def save_progress(progress: dict):
    """Save progress checkpoint."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)
 
 
def main():
    print("=" * 60)
    print("  Tata Motors Comments - Google Translate Pipeline")
    print("=" * 60)
 
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"\n✓ Loaded {len(raw_data):,} comments")
 
    progress = load_progress()
    if progress:
        print(f"✓ Resuming from checkpoint — {len(progress):,} already done")
 
    translator = GoogleTranslator(source="auto", target="en")
    results = dict(progress)  # id(str) -> {original, translated}
 
    total       = len(raw_data)
    translated  = 0
    skipped     = 0
    errors      = 0
 
    print(f"\nProcessing {total:,} comments...\n")
 
    for i, item in enumerate(raw_data):
        idx   = str(item[0])
        month = item[1]
        text  = item[2] if len(item) > 2 else ""
 
        if idx in results:
            continue
 
        if is_english(text):
            results[idx] = {
                "month":      month,
                "original":   text,
                "translated": text,   
                "was_translated": False,
            }
            skipped += 1
        else:
            try:
                eng = translator.translate(text)
                results[idx] = {
                    "month":      month,
                    "original":   text,
                    "translated": eng if eng else text,
                    "was_translated": True,
                }
                translated += 1
                time.sleep(SLEEP_SEC)
            except Exception as e:
                results[idx] = {
                    "month":      month,
                    "original":   text,
                    "translated": text,
                    "was_translated": False,
                }
                errors += 1
 
        # Progress display
        done = len(results)
        if done % 100 == 0 or done == total:
            pct = 100 * done / total
            print(f"  {done:>6}/{total}  ({pct:.1f}%)  "
                  f"translated={translated}  skipped={skipped}  errors={errors}",
                  flush=True)
            save_progress(results)
 
    save_progress(results)
 
    output = []
    for item in raw_data:
        idx = str(item[0])
        r   = results.get(idx, {})
        output.append({
            "id":              item[0],
            "month":           item[1],
            "original_text":   r.get("original",   item[2] if len(item) > 2 else ""),
            "translated_text": r.get("translated",  item[2] if len(item) > 2 else ""),
            "was_translated":  r.get("was_translated", False),
        })
 
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    fields = ["id", "month", "original_text", "translated_text", "was_translated"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
 
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
 
    was_translated_count = sum(1 for o in output if o["was_translated"])
    print(f"\n✅ Complete!")
    print(f"   {was_translated_count:,} comments translated")
    print(f"   {len(output) - was_translated_count:,} already English (kept as-is)")
    print(f"   Saved → {OUTPUT_JSON}")
    print(f"   Saved → {OUTPUT_CSV}")
    print(f"\n   Use 'translated_text' column for your ML model.\n")
 
 
if __name__ == "__main__":
    main()