## ROC-AUC analysis

Performs analysis to determine if cells are significantly more active during some given events of
interest. 

Currently works for data tables formatted for a particular experiment. Examples of this format can be found here: 

`ultra94:/media/core/core_operations/ImageAnalysisScratch/Schwarz/Cameron/ImagingData/`

Notebooks included:
* 0_EDA.Rmd: Exploratory data analysis
* 1_ROC.Rmd: Performs ROC analysis for each cell in one recording
* 2_ROC_multirecording.Rmd: Performs ROC analysis for all cells in all recordings
* 3_ROC_multi_anlaysis.Rmd: Analyzes these results across recordings by matching cells with same ID