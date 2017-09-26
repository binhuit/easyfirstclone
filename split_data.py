data = open("data/fold_train_1.conll",'r')
output = open("data/train_1000","w")
sents = []
sent = []
for line in data:
    if line == "\n" or line == "\r\n":
        sents.append(sent)
        sent = []
    else:
        sent.append(line)

for sent in sents[:1000]:
    for tok in sent:
        output.write(tok)
    output.write('\n')


