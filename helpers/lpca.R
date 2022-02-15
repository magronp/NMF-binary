
logisticPCA <- function(Y, mask_leftout, k, maxit){

  start_time <- Sys.time()
  Y.train <- as.matrix(Y)
  is.na(Y.train) <- as.logical(mask_leftout)
  logpca_model <- logisticPCA::logisticSVD(Y.train, k, max_iters = maxit)
  end_time <- Sys.time()
  tot_time <- end_time - start_time
  W <- logpca_model$A
  H <- logpca_model$B
  Y_hat <- fitted(logpca_model, type = "response")

  return(list("W" = W, "H" = H, "Y_hat" = Y_hat, "tot_time" = tot_time))
}