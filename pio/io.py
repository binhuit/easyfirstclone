from yutils import tokenize_blanks

def to_tok(line):
   if line[4]=="_": line[4]=line[3]
   return {"parent": int(line[-4]),
           "prel"  : line[-3],
           "form"  : line[1],
           "lem"  : line[2],
           "id"    : int(line[0]),
           "tag"   : line[4],
           "ctag"   : line[3],
           "morph" : line[-5].split("|"),
           "extra" :  line[-1],
           }

def conll_to_sents(file):
    """
    yield sentences from corpus, each sentence is a list of tokens
    :param file: corpus file
    :return: yield sentences
    """
    for sent in tokenize_blanks(file):
        yield [to_tok(l) for l in sent]


