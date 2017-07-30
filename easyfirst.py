from moduleloader import load_module
from deps import DependenciesCollection
from common import ROOT
from collections import defaultdict
from ml.ml import MulticlassModel, MultitronParameters
import os
import sys

class Oracle:  # {{{
    def __init__(self):
        self.sent = None
        self.childs = defaultdict(set)

    def allow_connection(self, sent, deps, parent, child):
        if self.sent != sent:
            self.sent = sent
            self.childs = defaultdict(set)
            for tok in sent:
                self.childs[tok['parent']].add((tok['parent'], tok['id']))

        if child['parent'] != parent['id']:
            return False
        # if child didn't collect all it's childs, it can't connect.
        if len(self.childs[child['id']] - deps.deps) > 0:
            return False
        return True
        # }}}


class Parser:
    def __init__(self, scorer, featExt, oracle=None):
        self.scorer = scorer
        self.featExt = featExt
        self.oracle = oracle

    def train(self, sent):
        updates = 0
        sent = [ROOT] + sent
        self.scorer.tick()
        deps = DependenciesCollection()
        parsed = sent[:]
        fcache = {}
        scache = {}
        while len(parsed) > 1:
            best = -999999
            best_pair = None
            scored = []
            for i, (tok1, tok2) in enumerate(zip(parsed, parsed[1:])):
                tid = tok1["id"]
                if tid in fcache:
                    feats = fcache[tid]
                else:
                    feats = self.featExt.extract(parsed, deps, i, sent)
                    fcache[tid] = feats
                if tid in scache:
                    s1, s2 = scache[tid]
                else:
                    scores = self.scorer.get_scores(feats)
                    s1 = scores[0]
                    s2 = scores[1]
                scored.append((s1, 0, feats, tok1, tok2))
                scored.append((s2, 1, feats, tok2, tok1))
            # xap xep tu lon den nho, -s
            scored = sorted(scored, key=lambda (s, cls, f, t1, t2): -s)
            s, cls, f, c, p = scored[0]

            if self.oracle.allow_connection(sent, deps, p, c):
                # remove the neighbours of parent from the cache
                i = parsed.index(p)
                frm = i - 4
                to = i + 4
                if frm < 0: frm = 0
                if to >= len(parsed): to = len(parsed) - 1
                for tok in parsed[frm:to]:
                    try:
                        del fcache[tok['id']]
                        del scache[tok['id']]
                    except:
                        pass
                ###
                deps.add(p, c)
                parsed = [x for x in parsed if x != c]
            else:
                scache = {}  # clear the cache -- numbers changed..
                # find best allowable pair
                for s, gcls, gf, gc, gp in scored[1:]:
                    if self.oracle.allow_connection(sent, deps, gp, gc):
                        break

                self.scorer.add(f, cls, -1)
                self.scorer.add(gf, gcls, 1)

                updates += 1
                if updates > 200:
                    print "STUCK, probably because of incomplete feature set"
                    print " ".join([x['form'] for x in sent])
                    print " ".join([x['form'] for x in parsed])
                    return


class Model:
    def __init__(self, featuresFile, weightFile, iter=None):
        self._featuresFile = featuresFile
        self._weightFile = weightFile
        self._iter = iter

        featuresModule = load_module(featuresFile)
        self.fext = featuresModule.FeaturesExtractor()

    def save(self, filename):
        fh = file(filename, "w")
        fh.write("%s\n%s\n" % (self._featuresFile, self._weightFile))
        fh.close()

    @classmethod
    def load(cls, filename, iter=19):
        """
        
        :param filename: 
        :param iter: 
        :return: Model object 
        """
        lines = file(filename).readlines()
        # dirname = os.path.dirname(filename)
        dirname = os.path.curdir
        featuresFile = os.path.join(dirname, lines[0].strip())
        weightFile = os.path.join(dirname, lines[1].strip())
        return cls(featuresFile, weightFile, iter)

    def featureExtractor(self):
        return self.fext

    def weightsFile(self, iter):
        if iter is None: iter = self._iter
        return "%s.%s" % (self._weightFile, iter)

def train(sents, model, dev=None, ITERS=20, save_every=None):
    fext = model.featureExtractor()
    oracle = Oracle()
    scorer = MultitronParameters(2)
    parser = Parser(scorer, fext, oracle)
    for ITER in xrange(1, ITERS + 1):
        print "Iteration ",ITER,"[",
        for i, sent in enumerate(sents):
            if i % 100 == 0:
                print i
                sys.stdout.flush()
            parser.train(sent)
        print "]"
        if save_every and (ITER % save_every == 0):
            print "saving weights at iter", ITER
            parser.scorer.dump_fin(file(model.weightsFile(ITER), "w"))
            # if dev:
            #     print "testing dev"
            #     print "\nscore: %s" % (test(dev, model, ITER, quiet=True),)
        parser.scorer.dump_fin(file(model.weightsFile("FINAL"), "w"))




