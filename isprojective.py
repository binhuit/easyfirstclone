
def is_projective(sent):
    """
    check if sentence is projective
    :param sent: 
    :return: bool
    """
    spans = []
    for token in sent:
        s = tuple(sorted([token["id"],token["parent"]]))
        spans.append(s)
    spans = sorted(spans)
    for l,h in spans:
        for l1, h1 in spans:
            if (l,h) == (l1,h1): continue
            if l< l1 < h and h1 > h:
                return False
    return True

