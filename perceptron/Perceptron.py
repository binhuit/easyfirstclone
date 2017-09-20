import numpy as np
import pickle
from ml.ml import MulticlassModel, MultitronParameters
import os
class Hash:
    def __init__(self):
        self.hash_table = {}
        self.count = 0

    def get_and_add(self,feature):
        if feature in self.hash_table:
            return self.hash_table[feature]
        else:
            self.hash_table[feature] = self.count
            self.count += 1
            return self.hash_table[feature]

    def get_value(self,feature):
        if feature in self.hash_table:
            return self.hash_table[feature]
        else: return None
    def get_hash(self):
        return self.hash_table

class Perceptron:
    def __init__(self, n_class,w_size):
        self.n = n_class
        self.w_size = w_size
        self.hash = Hash()
        # self._paramaters = np.zeros((self.n,self.w_size))
        # self._last_updated = np.zeros((self.n,self.w_size))
        # self._a = np.zeros((self.n,self.w_size))
        self.now = 0
        self.feat_last_index = -1
        sample = [[] for i in xrange(self.n)]
        self._paramaters = np.asarray(sample)
        self._last_updated = np.asarray(sample)
        self._a = np.asarray(sample)

    def tick(self):
        self.now += 1

    def hash_vectorizer(self, features):
        feat_dict = {}
        x = []
        for f in features:
            h = self.hash.get_and_add(f)
            feat_index = h % self.w_size
            try:
                feat_dict[feat_index] += 1
            except KeyError:
                feat_dict[feat_index] = 1
        x = [(k, v) for k, v in feat_dict.iteritems()]
        indexes, values = zip(*x)
        indexes = np.asarray(indexes)
        values = np.asarray(values)
        return indexes,values

    def vectorizer(self, features):
        new_feat = 0
        feat_dict = {}
        for f in features:
            h = self.hash.get_and_add(f)
            if h > self.feat_last_index:
                new_feat += 1
            try:
                feat_dict[h] += 1
            except KeyError:
                feat_dict[h] = 1
        if new_feat > 0:
            extended_paramater = np.zeros((self.n,new_feat))
            self._paramaters = np.concatenate((self._paramaters,extended_paramater),axis=1)
            self._a = np.concatenate((self._a,extended_paramater),axis=1)
            extended_paramater.fill(self.now)
            self._last_updated = np.concatenate((self._last_updated,extended_paramater),axis=1)
            self.feat_last_index += new_feat
        x = [(k, v) for k, v in feat_dict.iteritems()]
        indexes, values = zip(*x)
        indexes = np.asarray(indexes)
        values = np.asarray(values)
        return indexes,values

    def get_scores(self, features):
        indexes, values = self.vectorizer(features)
        w = np.take(self._paramaters,indexes,1)
        scores = np.dot(w,values).tolist()
        return scores

    def add(self,features, cls, amount):
        indexes, values = self.vectorizer(features)
        values = amount*values
        # update a
        # calculate voting
        voting = self.now - self._last_updated[cls,:][indexes]
        self._a[cls,:][indexes] += voting*self._paramaters[cls,:][indexes]
        self._paramaters[cls,:][indexes] += values
        self._last_updated[cls,:][indexes] = self.now

    def dump_fin(self,out):
        # o = {'hash':self.hash,'model':self}
        # pickle.dump(o,out)
        weight = (self.now - self._last_updated)*self._paramaters/self.now
        out.write(str(self.n)+'\n')
        out.write(str(self.feat_last_index+1)+'\n')
        feat_hash = self.hash.get_hash()
        for key, value in feat_hash.items():
            column = [str(w) for w in weight[:,value].tolist()]
            line = [key] + column
            line = ' '.join(line)
            line += '\n'
            out.write(line)


    @classmethod
    def load(cls,fname):
        package = pickle.load(file(fname))
        hash = package['hash']
        model = package['model']
        perceptron = cls(model.n, model.w_size)
        perceptron.hash = hash
        perceptron._paramaters = (model.now - model._last_updated)*model._paramaters/model.now
        return perceptron

class MultiClass:
    def __init__(self):
        self._paramaters = None
        self.hash = None

    def get_scores(self,features):
        indexes, values = self.vectorizer(features)
        w = np.take(self._paramaters,indexes,1)
        scores = np.dot(w,values).tolist()
        return scores

    def vectorizer(self, features):
        feat_dict = {}
        for f in features:
            h = self.hash.get_value(f)
            if h:
                feat_dict[h] = 1
        x = [(k, v) for k, v in feat_dict.iteritems()]
        indexes, values = zip(*x)
        indexes = np.asarray(indexes)
        values = np.asarray(values)
        return indexes,values

    @classmethod
    def load(cls,fname):
        package = pickle.load(file(fname))
        hash = package['hash']
        model = package['model']
        classifier = cls()
        classifier.hash = hash
        classifier._paramaters = (model.now - model._last_updated)*model._paramaters/model.now
        return classifier
