setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

test.gbm <- function() {
  browser()
  x <- h2o.importFile(locate("bigdata/laptop/jira/reproducibility_issue.csv.zip"))

  # AUC thresholds run with 8 nodes from another machine.  Test results should match this.
  seeds = c(-8736384372227723979, 8568380275437571355, 6262970961823075204)
  trees = c(10, 20, 30)
  csvFiles = c(locate('bigdata/laptop/jira/threshold_n8_10t_2d_sn8736384372227723979.csv'),
               locate('bigdata/laptop/jira/threshold_n8_20t_2d_s8568380275437571355.csv'),
               locate('bigdata/laptop/jira/threshold_n8_30t_2d_sn6262970961823075204.csv'))
  index = sample(c(1:length(seeds)))[1]

  # Train Model
  gbm_v1 <- TrainGBM(x, "gbm_2T_3D_n3.hex", seeds[index], trees[index])
  auc_gbm = gbm_v1@model$training_metrics@metrics$thresholds_and_metric_scores$threshold
  auc_other = read.csv(csvFiles[index], header=T)[,2]
  expect_equal(auc_gbm, auc_other)

}

# borrowed from Megan K
TrainGBM <- function(data, model_id, seedNum, nTrees) {
  h2o.gbm(x=2:365, y='response', training_frame = data,
          distribution = "bernoulli", ntrees = nTrees, seed = seedNum, max_depth = 2, min_rows = 7,
          model_id= model_id, score_tree_interval=nTrees
  )
}

doTest("GBM reproducibility test", test.gbm)
