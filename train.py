import sys

from optparse import OptionParser


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
DEV_FILE = args[1] if args[1] else None
MODEL = opts.model_file
FEATURES = opts.features_file
