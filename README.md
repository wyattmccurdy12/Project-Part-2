# Final Project

In this project, I run a baseline from NailP based on their 2022 results in CLEF eRisk. 
Following up, I make some modifications to the baseline to find out whether results improve.  

The project is currently split into a main.py file for the first part of the run - data collation/preprocessing - and a notebook for calculations that require GPUs - the embedding and cosine similarity parts. (SimilarityMetrics_and_ModelEvaluation.ipynb)

## CSV Files

`ranking_augmented_data.csv`: This is the csv with all the ranking scores (cosine sim) from the baseline model run - all-Mini-LM. 

`augmented_answer_sets.csv`: Answer sets generated by the NailP team and provided in their publication. The size of the sample answer sets from the original BDI is increased tenfold.

## TSV files

Files for the second run will be in TSV format - it is generally better practice to store files in TSV format for these tasks.

`dbue_results.tsv`: "distilbert-base-uncased-emotion" results. This is from the non-base model run. It is a trec-formatted output.

## Notebooks

`SimilarityMetrics_and_ModelEvaluation.ipynb`: Read preprocessed data from main.py and feed into process outlined by NailP team in their paper: NailP at eRisk 2023: Search for Symptoms of Depression.

`Post_Baseline_Run.ipynb`: The above notebook, but modified to produce results from distillbert trained on emotional data.

## Python (.py) Files

`main.py`: Incomplete. Preprocessing steps for trec-formatted input data.

## Directories
`task1`: The folder provided by the team at CLEF eRisk containing training and evaluation data.