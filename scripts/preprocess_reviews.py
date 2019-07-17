import gzip
import json
import re
import numpy as np
from random import Random

rand = Random(12345)

stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

reviews = []
with gzip.GzipFile('../data/reviews_Kindle_Store_5.json.gz', 'r') as fp:
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

reviews = [(r['review_indices'], r['overall']) for r in reviews]

# Remove 3 star reviews to simplify problem
reviews = list(filter(lambda r: r[1] != 3.0, reviews))

num_positive = sum(1 for r in reviews if r[1] > 3)
num_negative = len(reviews) - num_positive
    
if False:
  # Balance out the dataset so we have equal positive/negative
  assert num_negative < num_positive
  # sort from negative to positive
  reviews.sort(key = lambda r:r[1])
  # grab the least favorable and the most favorable equally
  reviews = reviews[0:num_negative] + reviews[-num_negative:]
  num_positive = sum(1 for r in reviews if r[1] > 3)
  num_negative = len(reviews) - num_positive

print("Total of %d reviews (%d%% positive)" % (len(reviews), num_positive * 100 / len(reviews)))

rand.shuffle(reviews)
vocab_size = len(top10k_indices)
ident_topk = np.eye(vocab_size)
# Use uint8 for memory efficiency.  When it gets multiplied it will get casted up
x_all = np.zeros((len(reviews), vocab_size), dtype=np.uint8)
# Since y_all is small, we keep it at a float32 since it can be error prone to manipulate without casting (it hit me)
y_all = np.zeros((len(reviews),), dtype=np.float32)
for i, r in enumerate(reviews):
  for w in r[0]:
    x_all[i, w] = 1
  # Make it a binary classification (positive/negative review)
  y_all[i] = 1 if r[1] > 3 else 0


for config in [('all', int(len(reviews)*.9), len(reviews)), ('200k', int(200000*.9), 200000), ('100k', int(100000*.9), 100000), ('50k', int(50000*.9), 50000)]:
  filename = config[0] + '_reviews_binary_10k_vocab.npz'
  print("Saving %s" % filename)
  np.savez_compressed(filename,
                      vocab=top10k_indices,
                      x_train=x_all[0:config[1]],
                      y_train=y_all[0:config[1]],
                      x_test=x_all[config[1]:config[2]],
                      y_test=y_all[config[1]:config[2]])
