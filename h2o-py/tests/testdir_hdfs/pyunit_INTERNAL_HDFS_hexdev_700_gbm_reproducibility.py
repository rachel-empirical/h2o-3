import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import numpy as np
import time

# HEXDEV-700: GBM reproducibility issue.
def gbm_reproducibility():

  # Check if we are running inside the H2O network by seeing if we can touch
  # the namenode.
  hadoop_namenode_is_accessible = pyunit_utils.hadoop_namenode_is_accessible()

  if hadoop_namenode_is_accessible:
    hdfs_name_node = pyunit_utils.hadoop_namenode()

    # import auc from 13 node runs
    hdfs_csv_file = "/datasets/threshold_n8_10t_2d_sn8736384372227723979.csv"
    url_csv = "hdfs://{0}{1}".format(hdfs_name_node, hdfs_csv_file)
    auc_13node = h2o.import_file(url_csv)
    auc_n13 = np.transpose(auc_13node[1].as_data_frame().values)[0]  # grab it as a list

    #import data frame
    hdfs_csv_file = "/datasets/reproducibility_issue.csv.zip"
    url_csv = "hdfs://{0}{1}".format(hdfs_name_node, hdfs_csv_file)
    h2oframe_csv = h2o.import_file(url_csv)

    # build gbm model
    t0 = time.time()
    gbm = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=50, seed= 987654321, max_depth = 4,
                                       min_rows = 7, score_tree_internal=50)
    gbm.train(x=list(range(2,365)), y="response", training_frame=h2oframe_csv)
    t1 = time.time()-t0
    print("Run time is {0}".format(t1))
    auc_h2o = pyunit_utils.extract_from_twoDimTable(gbm._model_json['output']['training_metrics']._metric_json['thresholds_and_metric_scores'], 'threshold', takeFirst=False)
    pyunit_utils.equal_two_arrays(auc_n13, auc_h2o, 1e-10, True)  # compare two thresholds and they should equal
  else:
    raise EnvironmentError


if __name__ == "__main__":
  pyunit_utils.standalone_test(gbm_reproducibility)
else:
  gbm_reproducibility()
