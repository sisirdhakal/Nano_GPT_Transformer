text = """
एकैसाथ दुई पुस्तक लिएर आएका छन् लेखक तथा पत्रकार सुधीर शर्मा । इतिहासदेखि सन् २००८ सम्मको नेपाल र चीनबीचको सांस्कृतिक, आर्थिक र राजनीतिक सम्बन्धका आधारमा ‘भिक्षु, व्यापार र विद्रोह’ किताब लेखका छन् । त्यस्तै नेपालमा गणतन्त्र स्थापनापछिको सम्बन्धका आयाममा ‘हिमालपारिको हुरी’ लेखका छन् ।


हिमालपारिको हुरीले विश्व व्यवस्थामा चीनको महत्त्वाकांक्षा र प्रभावका पृष्ठभूमिमा नेपालसँगको सम्बन्धलाई विवेचना गरेको छ । किताब विमोचनको पूर्वसन्ध्यामा कान्तिपुरका उमेश चौहान र किशोर दाहालले शर्मासँग संवाद गरेका छन्–
"""
import re
def convert_to_one_sentence(text):
    # Remove any extraneous newlines and extra spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally, remove specific punctuation (e.g., if you want to handle it differently)
    text = re.sub(r'[।?!]', '', text)  # Removing end punctuation marks
    text = text.replace('।', '.').replace('?', '.').replace('!', '.')

    # Remove any remaining extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

textData = convert_to_one_sentence(text)

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ---
vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx


# merges = {} # (int, int) -> int
# count = 0
# while True:
#   stats = get_stats(ids)
#   pair = max(stats, key=stats.get)
#   if stats[pair] == 1:
#     break
#   idx = 256 + count
#   print(f"merging {pair} into a new token {idx}")
#   ids = merge(ids, pair, idx)
#   merges[pair] = idx
#   count +=1

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
    # print(tokens)
  return tokens

name = 'शिशिर'
print(encode(name))
print(decode(encode(name)))

# wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe

# wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json

import os, json

with open('encoder.json', 'r') as f:
    encoder = json.load(f) # <--- ~equivalent to our "vocab"

with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = {tuple(merge_str.split()):256+i for i, merge_str in enumerate(bpe_data.split('\n')[1:-1])}
# ^---- ~equivalent to our "merges"

# encoder sample
# {'!': 0, '"': 1, '#': 2, '$': 3, '%': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9}

# bpe_merges sample
# [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'), ('i', 'n'), ('r', 'e'), ('o', 'n'), ('Ġt', 'he'), ('e', 'r'), ('Ġ', 's'), ('a', 't')]