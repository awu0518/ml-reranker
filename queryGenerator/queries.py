
queries = []
with open("queries.train.tsv", "r", encoding="utf-8") as f:
    for line in f:
        qid, query = line.strip().split(maxsplit=1)
        queries.append(query)
        if len(queries) == 500:
            break

with open("queriesSubset", "w", encoding="utf-8") as f:
            f.write("\n".join(queries))