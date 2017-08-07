from moduleloader import load_module
from deps import DependenciesCollection
from common import ROOT
from collections import defaultdict
from ml.ml import MulticlassModel, MultitronParameters
from itertools import izip,islice
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

    def parse(self, sent):  # {{{
        deps = DependenciesCollection()
        parsed = sent[:]
        parsed = [ROOT] + parsed
        sent = [ROOT] + sent
        scache = {}
        fe = self.featExt.extract
        gscore = self.scorer.get_scores
        lp = len(parsed)
        while lp > 1:
            # find best action
            _pairs = []
            for i, (tok1, tok2) in enumerate(izip(parsed, islice(parsed, 1, None))):
                tid = tok1['id']
                if tid in scache:
                    s1, s2 = scache[tid]
                else:
                    feats = fe(parsed, deps, i, sent)
                    scr = gscore(feats)
                    s1 = scr[0]
                    s2 = scr[1]
                    scache[tid] = s1, s2

                _pairs.append((s1, tok1, tok2, i + 1))
                _pairs.append((s2, tok2, tok1, i))

            best, c, p, locidx = max(_pairs)
            # remove the neighbours of parent from the cache
            i = locidx
            frm = i - 4
            to = i + 4
            if frm < 0: frm = 0
            if to >= lp: to = lp - 1
            for tok in parsed[frm:to]:
                try:
                    del scache[tok['id']]
                except:
                    pass
            # apply action
            deps.add(p, c)
            parsed.remove(c)
            lp -= 1
        return deps


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
                    scache[tid] = s1, s2
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

def parse(sents, model, iter="FINAL"):
    fext = model.featureExtractor()
    m = MulticlassModel(model.weightsFile(iter))
    parser = Parser(m, fext, Oracle())
    for sent in sents:
        deps = parser.parse(sent)
        sent = deps.annotate(sent)
        for tok in sent:
            print tok['id'], tok['form'], "_", tok['tag'], tok['tag'], "_", tok['pparent'], "_ _ _"
        print

def test(sents, model, iter="FINAL", quiet=False, ignore_punc=False):
    fext = model.featureExtractor()
    import time
    good = 0.0
    bad = 0.0
    complete = 0.0
    m = MulticlassModel(model.weightsFile(iter))
    start = time.time()
    parser = Parser(m, fext, Oracle())
    scores = []
    for sent in sents:
        sent_good = 0.0
        sent_bad = 0.0
        no_mistakes = True
        if not quiet:
            print "@@@", good / (good + bad + 1)
        deps = parser.parse(sent)
        sent = deps.annotate(sent)
        for tok in sent:
            if not quiet: print tok['id'], tok['form'], "_", tok['tag'], tok['tag'], "_", tok['pparent'], "_ _ _"
            if ignore_punc and tok['form'][0] in "'`,.-;:!?{}": continue
            if tok['parent'] == tok['pparent']:
                good += 1
                sent_good += 1
            else:
                bad += 1
                sent_bad += 1
                no_mistakes = False
        if not quiet: print
        if no_mistakes: complete += 1
        scores.append((sent_good / (sent_good + sent_bad)))

    if not quiet:
        print "time(seconds):", time.time() - start
        print "num sents:", len(sents)
        print "complete:", complete / len(sents)
        print "macro:", sum(scores) / len(scores)
        print "micro:", good / (good + bad)
    return good / (good + bad), complete / len(sents)






