# import os

# try:
#     import rpy2.robjects as robjects
# except OSError as e:
#     try:
#         import os
#         import platform
#         if ('Windows', 'Microsoft') in platform.system():
#             os.environ["R_HOME"] = 'D:/R/R-4.3.1/bin/x64'  # Your R version here 'R-4.0.3'
#             os.environ["PATH"] = "D:/R/R-4.3.1/bin/x64" + ";" + os.environ["PATH"]
#         import rpy2.robjects as robjects
#     except OSError:
#         raise(e)

# path = '.'

# def compute(x, y):
#     # print(os.getcwd())
#     all_scags = {}
#     r_source = robjects.r['source']
#     # r_source(os.path.join(path, '../../DRflow/metrics/get_scag.r'))
#     r_source(os.path.join('./get_scag.r'))
#     r_getname = robjects.globalenv['scags']
#     scags = r_getname(robjects.FloatVector(x), robjects.FloatVector(y))
#     all_scags['outlying'] = scags[0]
#     all_scags['skewed'] = scags[1]
#     all_scags['clumpy'] = scags[2]
#     all_scags['sparse'] = scags[3]
#     all_scags['striated'] = scags[4]
#     all_scags['convex'] = scags[5]
#     all_scags['skinny'] = scags[6]
#     all_scags['stringy'] = scags[7]
#     all_scags['monotonic'] = scags[8]
#     return all_scags

# # all_scags = compute(z_tsne[:, 0], z_tsne[:, 1])
# # print("all_scags: ", all_scags)

import os
import rpy2.robjects as robjects

path = '.'

def compute(x, y):
    # print(os.getcwd())
    all_scags = {}
    r_source = robjects.r['source']
    r_source(os.path.join('./get_scag.r'))
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