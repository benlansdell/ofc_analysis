source('./helper.R')

#Run EDA on all recordings
run_all_eda = function(wd = '/home/blansdel/projects/roc_cell_analysis', 
                       path = '/media/core/core_operations/ImageAnalysisScratch/Schwarz/Cameron/ImagingData/',
                       events = c('lspout_first', 'rspout_first')) {
  setwd(wd)
  data_files = list.files(path)
  for (data_file in data_files) {
    make_report('0_eda.Rmd', data_file = data_file, EOI = events)
  }
}

#Run ROC on all recordings
run_all_roc = function(wd = '/home/blansdel/projects/roc_cell_analysis', 
                       path = '/media/core/core_operations/ImageAnalysisScratch/Schwarz/Cameron/ImagingData/',
                       events = c('lspout_first', 'rspout_first')) {
  setwd(wd)
  data_files = list.files(path)
  for (data_file in data_files) {
    make_report('1_roc.Rmd', data_file = data_file, EOI = events)
  }
}