import os
import pandas as pd
from data_processing_utils import *
from data_processing_utils import DataPreProcessor
from data_processing_utils import LanguageProcessor
from data_processing_utils import EmbeddingProcessor
from data_processing_utils import PostProcessor
 ############

def main():
    '''
    DATA PREPROCESSING
    '''
    print("Starting the program...")

    # Define the top level directory - task 1 - contains training, results, testing data
    top_level_dir = '/home/wyatt.mccurdy/Github/Project-Part-2/task1/'
    training_data_dir = os.path.join(top_level_dir, 'training/t1_training/TRAINING DATA (2023 COLLECTION)/')
    print(f"Training data directory is set to: {training_data_dir}")

    # Define the paths to the TREC formatted files
    trec_formatted_files = os.path.join(training_data_dir, 'new_data/')
    dpp = DataPreProcessor(trec_formatted_files) # Initialize the data preprocessor

    # Tabulate and filter the data - filter on sentiment polarity and self reference (personal pronouns)
    print("Tabulating TREC data...")
    trec_df = dpp.trec_csv_from_dir(training_data_dir, trec_formatted_files, 'tabulated_unfiltered_trec.csv')
    trec_df = dpp.clean_text(trec_df, 'tabulated_cleaned_unfiltered_trec.csv')
    trec_df = dpp.persons_and_emotions(trec_df, 'tabulated_cleaned_emotionfiltered_trec.csv')

    # # Augmented answer sets from BDI q1-q21. 
    print("Processing augmented answer sets...")
    aug_answers_df = dpp.process_augmented_data('augmented_answer_sets.txt', 'augmented_answer_sets.csv')

    '''
    COSINE SIMILARITY RANK CALCULATION
    The goal is to take the cosine similarity between the embeddings of the TREC data and the augmented data.
    In the augmented data, there are 21 questions, and each question has 4 severity levels for the answers.
    Therefore, a higher cosine similarity between text from a user generated post and 
    embeddings associated with symptoms should give that post a higher ranking for that particular question.
    '''
    print("Starting the cosine similarity rank calculation...")

    # Modify slightly - add a question_num argument, do for all 21 questions, and save to 21 dataframes
    # sorted on similarity for the particular question
    ep = EmbeddingProcessor()
    cosine_similarity_dfs = []
    for i in range(1, 22):
        print(f"Calculating cosine similarity for question {i}...")
        cos_similarity_df = ep.similarity_sum_over_col(trec_df, aug_answers_df, question_num=i)
        print(f"Saving cosine similarity for question {i}...")
        cosine_similarity_dfs.append(cos_similarity_df)
    
    # Read in the rels table in order to evaluate the rankings - there are 2!
    training_rels_majority_path = os.path.join(training_data_dir, 'g_qrels_majority_2.csv')
    training_rels_consensus_path = os.path.join(training_data_dir, 'g_rels_consenso.csv')
    rels_majority_df = pd.read_csv(training_rels_majority_path)
    rels_consensus_df = pd.read_csv(training_rels_consensus_path)
    
    popr = PostProcessor()
    baseline_trec_table = popr.create_trec_table(cosine_similarity_dfs, 'baseline')

    # Compute the metrics for the TREC table
    metrics_majority = popr.compute_metrics(baseline_trec_table, rels_majority_df)
    metrics_consensus = popr.compute_metrics(baseline_trec_table, rels_consensus_df)




    # Save metrics to a csv file
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv('metrics.csv')
    print("Metrics saved to 'metrics.csv'.")


    print("Program completed.")

if __name__ == '__main__':
    main()