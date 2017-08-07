import sys
from pio import io
from easyfirst import test,parse,Model

from optparse import OptionParser

usage="""usage: %prog -m model [options] input_file """

parser = OptionParser(usage)
parser.add_option("-m","--model",dest="model_file")
parser.add_option("--iter",dest="iter",default="FINAL")
parser.add_option("-e","--eval",action="store_true",dest="eval",default=False)
parser.add_option("--nopunct",action="store_true",dest="ignore_punc",default=False)

opts, args = parser.parse_args()

if (not opts.model_file) or (len(args)!=1):
   parser.print_usage()
   sys.exit()

TEST_FILE = args[0]

model = Model.load("%s" % opts.model_file, opts.iter)

test_sents = [s for s in io.conll_to_sents(file(TEST_FILE))]

if opts.eval:
   test(test_sents,model,opts.iter,quiet=False,ignore_punc=opts.ignore_punc)
else:
   parse(test_sents,model,opts.iter)

