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
        print "adding",parent['id'],'->',child['id']

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
