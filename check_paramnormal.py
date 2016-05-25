import sys
import matplotlib
matplotlib.use('agg')

import paramnormal
status = paramnormal.test(*sys.argv[1:])
sys.exit(status)
