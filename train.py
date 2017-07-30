import sys

from optparse import OptionParser
from easyfirst import Model, train
from pio import io
from isprojective import is_projective
from deps import DependenciesCollection
usage = "usage: %prog -o model -f features [option] train_file [dev_file]"
parser = OptionParser(usage)
parser.add_option("-o","--model",dest="model_file")
parser.add_option("-f","--features",dest="features_file",default = None)
parser.add_option("--iter",type="int",dest = "iters",default = 20)
parser.add_option("--every",type="int",dest = "save_every",default = 1)

opts,args = parser.parse_args()

if len(args) < 1 or not(opts.model_file or opts.features_file):
    parser.print_help()
    sys.exit(1)

TRAIN_FILE = args[0]
DEV_FILE = args[1] if len(args)>1 else None
MODEL = opts.model_file
FEATURES = opts.features_file

model = Model(FEATURES,"%s.weights" % MODEL)
model.save("%s.model" % MODEL)

train_sents = list(io.conll_to_sents(file(TRAIN_FILE)))
print len(train_sents)
train_sents = [s for s in train_sents if is_projective(s)]
print len(train_sents)
dev = []
train(train_sents, model, dev, opts.iters,save_every=opts.save_every)