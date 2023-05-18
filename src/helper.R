library(foreach)
library(parallel)
library(doParallel)
library(data.table)
library(utils)
library(tools)
library(stringi)

compute_baseline_accuracy <- function(dt) { 
  total_rows <- nrow(dt)
  evnts <- unique(dt[,event])
  max_ct <- 0  
  for (evt in evnts) {
    nr <- nrow(dt[event == evt, ]) 
    if (nr > max_ct) {
      max_ct = nr
    }
  }
  return (max_ct/total_rows)    
}

plot_cm <- function(dt_zscored, evals, idx) {
  baseline_acc <- compute_baseline_accuracy(dt_zscored[`Trial Number` == idx])
  pval <- evals[idx]$p_value
  p <- plot_confusion_matrix(as.data.table(evals[idx]$`Confusion Matrix`[1]), 
                        rotate_y_text = FALSE, 
                        add_row_percentages = F, 
                        add_col_percentages = F,
                        add_normalized = F, rm_zero_percentages = F, rm_zero_text = F) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + ggtitle(paste('trial:', idx, '\nacc:', format(evals[idx]$`Overall Accuracy`, digits = 3),'\n baseline acc:', format(baseline_acc, digits = 3)))#,'\n p-value:', format(pval, digits = 3)))
  p  
}

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

make_datamatrix_dt = function(dt) {
  #' Take datatable and turn it into a wide table, with one column per cell
  #' 
  #' @description Take a data.table and turn it into a wide table, with one column per cell. Will z-score each column
  #' And will throw out cells that are too inactive. These are cells 
  #' 
  #' @param dt data.table. The recording data table
  #' @return The wide table, ready to run downstream analysis 

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