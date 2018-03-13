import json

# just some one time code, just ignore it
file_pre = 'svm_1'

with open(file_pre + '.json', 'r') as f:
    data = json.load(f)
    # scatter into little files
    chunks = [data[x:x+1000] for x in xrange(0, len(data), 1000)]

for chunk in chunks:
    with open(file_pre + '_split_' + str(chunks.index(chunk)) + '.json', 'w') as f:
        f.write(json.dumps({file_pre: chunk}))
