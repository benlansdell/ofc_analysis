library(foreach)
library(parallel)
library(doParallel)
library(data.table)
library(utils)

cycle = function (x,i) c(x[-(0:i)], x[0:i])

preprocess_dt = function(dt, events_of_interest) {
  ds = dt[,.(frame, CellID, `Trial Number`, event, dff)]
  for (evt in events_of_interest) {
    ds[, (evt) := as.integer(event == evt)]
  }
  ds[, frame_offset_event := frame - min(.SD[event %in% events_of_interest, frame]), by = `Trial Number`]
  return(ds)
}

compute_roc_par = function(per_cell_ds, events_of_interest, dt_roc, N = 1000, n_workers = 20) { 
  #Save each table as a separate file, so the parallel workers load this file, instead of each
  #loading the whole data table.
  
  #For each cell, save as a file
  #tmpdir = './tmp/'
  #dir.create(tmpdir)
  #for (idx in 1:length(per_cell_ds)) {
  #  fn = paste(tmpdir, 'cell_', idx, '.csv', sep = "")
  #  fwrite(per_cell_ds[[idx]], fn)
  #}

  #Create a cluster of workers
  cl <- makeCluster(n_workers)
  registerDoParallel(cl)
  
  cycle = function (x,i) c(x[-(0:i)], x[0:i])
  
  #Parallel
  null_rocs = foreach(idx = 1:length(per_cell_ds), .packages = c("data.table", "ROCR")) %dopar% {

    #fn = paste(tmpdir, 'cell_', idx, '.csv', sep = "")
    
    dss = per_cell_ds[[idx]]
    #This seems quite slow...
    #dss = fread(fn)
    
    cell = dss[1, CellID]
    print(cell)
    dt_null = data.table(CellID = cell)
    for (evt in events_of_interest) {
      aucs = c()
      for (j in 1:N) {
        #Perform shuffle of data...
        rand_idx = ceiling(runif(1, 0, length(dss$dff_zs)))
        shuffle = cycle(dss$dff_zs, rand_idx)
        auc = performance(prediction(shuffle, dss[,..evt]), "auc")@y.values[[1]]
        aucs = c(aucs, auc)
      }
      colname = paste(evt, 'AUC', sep = "_")
      this_auc = dt_roc[CellID == cell, ..colname]
      quantile = sum(aucs < this_auc[[1]])/N
      dt_null[,eval(paste('null_quantile', evt, sep = "_")) := quantile]
    }
    return(dt_null)

  }
  null_rocs = rbindlist(null_rocs)
  stopCluster(cl)
  return(null_rocs)
}