library(foreach)
library(parallel)
library(doParallel)
library(data.table)
library(utils)
library(tools)
library(stringi)

cycle = function (x,i) c(x[-(0:i)], x[0:i])

strip = function (x) {tools::file_path_sans_ext(basename(x))}

preprocess_dt = function(dt, events_of_interest) {
  #' Perform basis preprocessing of data table
  #' 
  #' @description Adds indicator variable to data table for each event of interest. 
  #' 
  #' @param dt data.table. The recording data table
  #' @param events_of_interest list. A list of strings containing the events of interest. 
  #' These are matched to the 'event' column in dt. 
  #' @return The processed data.table.

  ds = dt[,.(frame, CellID, `Trial Number`, event, dff, MatchID, Session, Animal)]
  for (evt in events_of_interest) {
    ds[, (evt) := as.integer(event == evt)]
  }
  ds[, frame_offset_event := frame - min(.SD[event %in% events_of_interest, frame]), by = `Trial Number`]
  return(ds)
}

compute_roc_par = function(per_cell_ds, events_of_interest, dt_roc, N = 1000, n_workers = 20) { 
  #' Compute statistical significance of ROC AUC for a list of cells, 
  #' using a parallel implementation for speed.
  #' 
  #' @description Given a list of data.tables, each housing data for a cell, generate a 
  #' null-distribution to determine if that ROC AUC is statistically significant.
  #' 
  #' @param per_cell_ds list of data.table. The recording data table, split into each cell
  #' @param events_of_interest list. A list of strings containing the events of interest. 
  #' These are matched to the 'event' column in dt. 
  #' @param dt_roc data.table. The data table housing the ROC AUC for each cell
  #' @param N int. Number of shuffles to perform to generate null-distribution (default 1000)
  #' @param n_workers int. Number of parallel workers to instantiate (default 20) 
  #' 
  #' @return data.table housing the quantile the true AUC lies in relation to the 
  #' generated null-distribution

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

compute_all_rocs = function(path, recordings, events_of_interest, N, alpha) { 
  #' For a set of recordings, run through entire ROC analysis
  #' 
  #' @description Preprocess, compute ROC AUC for each cell in a set of recordings, and determine
  #' its statistical significance. 
  #' 
  #' @param path string. Path to the recordings 
  #' @param recordings list of strings. List of recordings to analyze
  #' @param events_of_interest list. A list of strings containing the events of interest. 
  #' These are matched to the 'event' column in dt. 
  #' @param N int. Number of shuffles to perform to generate null-distribution (default 1000)
  #' @param alpha double. Threshold at which to call something statistically significant 
  #' 
  #' @return data.table housing results of the analysis for all recordings
  
  dt_roc = data.table(recording = character(), Animal = character(), Session = character(), 
                      Reward = character(), CellID = character(), MatchID = character())
  
  for (fn in recordings) {
    print(paste('Performing ROC analysis for', fn))
    
    dt = readRDS(paste(path, fn, sep = '')) #load
    ds = preprocess_dt(dt, events_of_interest) #process data table
    ds[, dff_zs := (dff - mean(dff))/sd(dff), by = `CellID`] #z-score
    
    roc_curves = hash()
    per_cell_ds = split(ds, f = ds$CellID)
    cells = dt[(`Trial Number` == 1) & (frame == 1), CellID]
    matches = dt[(`Trial Number` == 1) & (frame == 1), MatchID]
    animal = dt[1, Animal]
    session = dt[1, Session]
    reward = dt[1, Reward]
    
    #Compute AUC ROC    
    for (evt in events_of_interest) {
      for (dss in per_cell_ds) {
        p = performance(prediction(dss$dff_zs, dss[,..evt]), "tpr", "fpr") 
        q = performance(prediction(dss$dff_zs, dss[,..evt]), "auc") 
        roc_curves[paste(evt, dss[1,CellID])] = list(p, q)
      }
    }
    
    dtt = data.table(recording = fn, Animal = animal, Session = session, Reward = reward, 
                     CellID = cells, MatchID = matches)
    ### Generation of null-distributions for each cell
    for (evt in events_of_interest){
      aucs = sapply(cells, function(x) roc_curves[[paste(evt, x)]][[2]]@y.values[[1]])
      dtt[,eval(paste(evt, 'AUC', sep = "_")) := aucs]
    }
    
    null_rocs = compute_roc_par(per_cell_ds, events_of_interest, dtt, N)
    dtt = merge(dtt, null_rocs, by = 'CellID')
    
    for (evt in events_of_interest) {
      colname = paste('null_quantile', evt, sep = "_")
      dss = dtt[,colname, with = FALSE]
      dtt[,(paste(evt, 'pos_resp', sep='_')) := (dss > 1-alpha)]
      dtt[,(paste(evt, 'neg_resp', sep='_')) := (dss < alpha)]
    }
    dt_roc = rbind(dt_roc, dtt, fill = TRUE)
  }

  return(dt_roc)
}

make_report = function(source_file, data_file = NULL, EOI = NULL, params = NULL) {
  #' Run a notebook for a given set of parameters and save as a html report
  #' 
  #' @description Runs either of the EDA or ROC analysis notebooks for a given recording file
  #' and events of interest. 
  #' 
  #' @param source_file string. The Rmd notebook
  #' @param data_file string. The recording file to analyze
  #' @param EOI list. A list of strings containing the events of interest. 
  #' These are matched to the 'event' column in dt. 
  #' @param params list. Structure generated dynamically by the Rmd notebook in Rstudio. If neither EOI
  #' or data_file are provided, then this can be used instead. If neither data_file, EOI or params are
  #' provided, then the parameters in the yaml Rmd header are used. 
  #' 
  #' @return Nothing. Outputs a html file for the report in ./reports/ directory of the working dir.
  
  #If we provide both data file and EOI then just use these 
  if (is.null(data_file) | is.null(EOI)) {
    if (is.null(params)) {
      #If all arguments are missing, get the parameters from the source file
      yaml = rmarkdown::yaml_front_matter(source_file)
      params = yaml$params
    }
    #Else we can get the from the params structure
    data_file = params$fn
    EOI = params$events_of_interest
  }
  dir.create('./reports', showWarnings = FALSE)
  fn = strip(source_file)
  events = paste(EOI, collapse = '_')
  fn_out = paste('./reports/', fn, '-', strip(data_file), '-', events, '.html', sep = "")
  rmarkdown::render(source_file, output_file=fn_out, 
                    params = list(fn = data_file, events_of_interest = EOI),
                    envir = new.env())
}

save_table = function(table, params) {
  #' Saves a data table named according to the parameters for the current notebook.
  #' 
  #' @param table data.table. The data table to be saved
  #' @param params #' @param params list. Structure generated dynamically by the Rmd notebook in Rstudio
  #' 
  #' @return Nothing
  
  dir.create('./results', showWarnings = FALSE)
  data_file = params$fn
  events = paste(params$events_of_interest, collapse = '_')
  fn_out = paste('./results/ROC_', strip(data_file), '-', events, '.csv', sep = "")
  fwrite(table, fn_out)
  
}