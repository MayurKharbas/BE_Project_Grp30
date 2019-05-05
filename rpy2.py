import rpy2.robjects as robjects
#
r_source = robjects.r['source']
r_source(‘cdr_cluster_analysis.R’)
#
print ‘r script finished running’
