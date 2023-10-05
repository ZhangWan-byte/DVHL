__author__ = 'avp'
"""
The following script calls an R script to compute scagnostics.
"""
import os
import rpy2.robjects as robjects

path = '.'

def compute(x, y):
    # print(os.getcwd())
    all_scags = {}
    r_source = robjects.r['source']
    # r_source(os.path.join(path, '../../DRflow/metrics/get_scag.r'))
    r_source(os.path.join('./metrics/get_scag.r'))
    print(os.path.join('./metrics/get_scag.r'))
    r_getname = robjects.globalenv['scags']
    scags = r_getname(robjects.FloatVector(x), robjects.FloatVector(y))
    all_scags['outlying'] = scags[0]
    all_scags['skewed'] = scags[1]
    all_scags['clumpy'] = scags[2]
    all_scags['sparse'] = scags[3]
    all_scags['striated'] = scags[4]
    all_scags['convex'] = scags[5]
    all_scags['skinny'] = scags[6]
    all_scags['stringy'] = scags[7]
    all_scags['monotonic'] = scags[8]
    return all_scags
