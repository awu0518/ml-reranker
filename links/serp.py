import requests
import json
import os
from dotenv import load_dotenv
from time import sleep

load_dotenv()
URL = "https://google.serper.dev/search"
API_KEY = os.getenv("API_KEY")

headers = {
  'X-API-KEY': API_KEY,
  'Content-Type': 'application/json'
}

queries = []
with open("queriesSubset", "r", encoding="utf-8") as f:
    for line in f:
        qid, query = line.strip().split("\t", 1)
        queries.append((qid, query))

print(f"Loaded {len(queries)} queries")

OUTPUT_PATH = "serp_links.jsonl"

with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
    for qid, query in queries:
        print(f"\nProcessing qid {qid}, query: {query}")
        seen_urls = set()
        rank = 0

        for page in range(1, 6):
            try:
                response = requests.request("POST", URL, headers=headers, json={"q": query, "page": page})
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"  Error on qid={qid}, page={page}: {e}")
                continue

            organic = data.get("organic", [])
            if not organic:
                print(f"  No organic results for page {page}")
                continue

            for i, r in enumerate(organic):
                url = r.get("link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                rank += 1

                record = {
                    "qid": qid,
                    "query": query,
                    "page": page,
                    "rank": rank,
                    "url": url,
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", "")
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # be nice to the API
        sleep(1)

print(f"\nDone. All links written to {OUTPUT_PATH}")
