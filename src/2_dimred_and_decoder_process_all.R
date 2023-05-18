source('./helper.R')

#Run decoder on all recordings
run_all_decoder = function(wd = '/home/blansdel/projects/roc_cell_analysis', 
                       path = '/media/core/core_operations/ImageAnalysisScratch/Schwarz/Cameron/ImagingData/') {
  setwd(wd)
  data_files = list.files(path)
  for (data_file in data_files) {
    make_report('1_dimred_and_decoder_eventstartsonly.Rmd', fn = data_file, path = path)
  }
}

run_all_decoder()