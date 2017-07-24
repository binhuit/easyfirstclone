
def tokenize_blanks(file):
    """
    Yield a sentence in the corpus, each token in the sentences is a line in the corpus file.
    :param file: corpus file
    :return: yield a stack contain a sentence
    """
    stack = []
    for line in file:
        line = line.strip().split()
        if not line:
            yield stack
            stack = []
        else:
            stack.append(line)
    if stack: yield stack


