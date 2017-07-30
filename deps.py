from collections import defaultdict

class DependenciesCollection:
    def __init__(self):
        self.deps = set()
        self._left_child = {}
        self._right_child = {}
        self._parents = {}
        self._childs = defaultdict(list)

    def add(self, parent, child):
        self.deps.add((parent['id'], child['id']))
        self._parents[child['id']] = parent
        self._childs[parent['id']].append(child)

        lc = self.left_child(parent)
        if (not lc) or child['id'] < lc['id']:
            self._left_child[parent['id']] = child

        rc = self.right_child(parent)
        if (not rc) or child['id'] > rc['id']:
            self._right_child[parent['id']] = child
        # print "adding",parent['id'],'->',child['id']

    def remove(self, parent, child):
        # kiem tra xem co trong deps khong
        if (parent, child) not in self.deps:
            raise Exception("No arcs in dep set")
        else:
            pid = parent['id']
            cid = child['id']
            del self._parents[cid]
            self._childs[pid].remove(child)
            children = sorted(self._childs[parent['id']]) # Right
            if children:
                if child == self.left_child(parent):
                    if children[0] < parent['id']:
                        self._left_child[parent['id']] = children[0]
                    else:
                        del self._left_child[parent['id']]
                elif child == self.right_child(parent):
                    if children[-1] > parent['id']:
                        self._right_child[parent['id']] = children[-1]
                    else:
                        del self._right_child[parent['id']]

    def left_child(self, tok):
        """
        Get right most child of the token
        :param tok: 
        :return: if the token has left-most child then return it, else return None
        """
        if tok is None: return None
        return self._left_child.get(tok['id'], None)

    def right_child(self, tok):
        """
        Get right most child of the token
        :param tok: 
        :return: if token has right-most child then return it, else return None
        """
        if tok is None: return None
        return self._right_child.get(tok['id'], None)

    def children(self, parent):
        if parent is None: return None
        return self._childs[parent['id']]

    def get_depth(self,tok):
        if not self.children(tok):
            return 1
        else:
            return max([self.get_depth(c) for c in self.children(tok)]) + 1

    def sibling(self, token, i=0):
        if not token: return None
        parent = self._parents.get(token["id"],None)
        if parent: parent = parent["id"]
        self._childs[parent].sort(key = lambda b:b["id"])
        siblings = self.children(parent)
        index = siblings.index(token)
        if 0 < (index+i) < len(siblings):
            return siblings[index+i]
        else:
            return None

    def span(self, tok):
        return self.right_border(tok) - self.left_border(tok)

    def parent(self, tok):
        return self._parents[tok['id']] if tok['id'] in self._parents else None

    def right_border(self, tok):
        r = self.right_child(tok)
        if not r:
            return int(tok['id'])
        else:
            return self.right_border(r)

    def left_border(self, tok):
        l = self.left_child(tok)
        if not l:
            return int(tok['id'])
        else:
            return self.left_border(l)



