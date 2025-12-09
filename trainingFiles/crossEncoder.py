import json
import os
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# If you have GPU: CrossEncoder(MODEL_NAME, device="cuda")
model = CrossEncoder(MODEL_NAME)

def getBestPassage(file_obj, model: CrossEncoder):
    passages = []
    first_obj = None

    for line in file_obj:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        if first_obj is None:
            first_obj = obj 

        passages.append({
            "passage_id": obj["passage_id"],
            "text": obj["passage"],
            "search_rank": obj["search_rank"],
            "url": obj["url"],
        })

    if not passages or first_obj is None:
        return None

    query = first_obj["query"]

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, p["text"]) for p in passages]

    # Cross-encoder scores: higher = more relevant
    scores = model.predict(pairs)  # returns list/np.array of floats

    # best passage index
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_passage = passages[best_idx]

    result = {
        "query_id": first_obj["query_id"],
        "query": query,
        "best_passage_id": best_passage["passage_id"],
        "best_passage": best_passage["text"],
        "cross_encoder_score": float(scores[best_idx]),
        "search_rank": best_passage["search_rank"],
        "url": best_passage["url"],
    }

    return result


# Make sure output directory exists
os.makedirs("bestPassagesCE", exist_ok=True)

for subfolder in os.listdir("passages"):
    subfolder_path = os.path.join("passages", subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    print("Processing doc:", subfolder)

    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            best = getBestPassage(f, model)

        if not best:
            print(f"ERROR ON {subfolder}/{filename}")
            continue

        out_path = os.path.join("bestPassagesCE", f"{subfolder}.jsonl")
        with open(out_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(best, ensure_ascii=False) + '\n')
