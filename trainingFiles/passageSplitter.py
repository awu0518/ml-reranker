import json
import os

for filename in os.listdir("queries"):
    with open(f"queries/{filename}", 'r',  encoding="utf-8") as f:
        os.makedirs(f"passages/{filename.split('.')[0]}", exist_ok=True)

        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"Skipping empty line {i} in file {filename}")
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Bad JSON at line {i}, file {filename}: {e}\nLine content: {line!r}")
                continue

            with open(f"passages/{filename.split('.')[0]}/{obj['search_rank']}.jsonl", "a", encoding="utf-8") as writeFile:
                text = obj['passage']
                counter = 1
                for j in range(0, len(text) - 200, 100):
                    doc = {
                        "query_id": obj["query_id"],
                        "query": obj["query"],
                        "search_rank": obj["search_rank"],
                        "passage": text[j:j+200],
                        "passage_id": f"{obj['passage_id']}_{counter}",
                        "url": obj["url"]
                    }

                    writeFile.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    counter += 1