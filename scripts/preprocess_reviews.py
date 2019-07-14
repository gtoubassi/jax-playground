import gzip
import json
import re

reviews = []
with gzip.GzipFile('reviews_Kindle_Store_5.json.gz', 'r') as fp:
  for line in fp:
    reviews.append(json.loads(line))

# lower case and remove non alphanumeric
for r in reviews:
  r['reviewText'] = re.sub('[^0-9a-z ]', '', r['reviewText'].lower())

# compute word counts
wordcounts = {}
for r in reviews:
  for w in r['reviewText'].split():
    wordcounts[w] = wordcounts.get(w, 0) + 1

wc = list(zip(wordcounts.keys(), wordcounts.values()))
sorted_wc = sorted(wc, key=lambda tup: tup[1], reverse=True)
top10k = list(list(zip(*sorted_wc[0:10000]))[0])
# zero reserved for UNK
top10k_indices = {k: v + 1 for v, k in enumerate(top10k)}
top10k_indices['<UNK>'] = 0

for r in reviews:
  r['review_indices'] = list(map(lambda w: top10k_indices.get(w, 0), r['reviewText'].split()))

processed_reviews = [(r['review_indices'], r['overall']) for r in reviews]
with open('all_reviews10k.json', 'w') as fp:
  json.dump((top10k_indices, processed_reviews), fp)

with open('50k_reviews10k.json', 'w') as fp:
  json.dump((top10k_indices, processed_reviews[0:50000]), fp)
