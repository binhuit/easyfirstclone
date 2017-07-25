from moduleloader import load_module
import os

class Model:
    def __init__(self, featuresFile, weightFile, iter = None):
        self._featuresFile = featuresFile
        self._weightFile = weightFile
        self._iter = iter

        featuresModule = load_module(featuresFile)
        self.fext = featuresModule.FeaturesExtractor()


    def save(self, filename):
        fh = file(filename,"w")
        fh.write("%s\n%s\n" % (self._featuresFile, self._weightFile))
        fh.close()

    @classmethod
    def load(cls, filename,iter = 19):
        """
        
        :param filename: 
        :param iter: 
        :return: Model object 
        """
        lines = file(filename).readlines()
        # dirname = os.path.dirname(filename)
        dirname = os.path.curdir
        featuresFile = os.path.join(dirname,lines[0].strip())
        weightFile = os.path.join(dirname,lines[1].strip())
        return cls(featuresFile, weightFile, iter)

    def featureExtractor(self):
        return self.fext

    def weightFile(self,iter):
        if iter is None: iter = self._iter
        return "%s.%s" % (self._weightFile,iter)




