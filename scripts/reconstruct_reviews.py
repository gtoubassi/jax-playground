import json

with open('processed_reviews10k.json', 'r') as fp:
  top10k, reviews = json.load(fp)
  for line in fp:
    reviews.append(json.loads(line))

top10k_decoder = dict((v,k) for k,v in top10k.items())

# lower case and remove non alphanumeric
for r in reviews:
  words = list(map(lambda wi: top10k_decoder[wi], r[0]))
  print(r[1], ' '.join(words))


