import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing_utils import *
from data_processing_utils import DataPreProcessor as dpp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score
from tqdm import tqdm


def main():

    '''
    DATA PREPROCESSING
    '''
    # Define the paths to the majority and consensus data
    # training_qrels_majority_2 = os.path.join(training_data_dir, 'g_qrels_majority_2.csv')
    # training_rels_consenso_path = os.path.join(training_data_dir, 'g_rels_consenso.csv')

    print("Starting the program...")

    # Define the top level directory - task 1 - contains training, results, testing data
    top_level_dir = '/home/wyatt.mccurdy/Github/Project-Part-2/task1/'
    training_data_dir = os.path.join(top_level_dir, 'training/t1_training/TRAINING DATA (2023 COLLECTION)/')
    print(f"Training data directory is set to: {training_data_dir}")

    # Define the paths to the TREC formatted files
    trec_formatted_files = os.path.join(training_data_dir, 'new_data/')

    print("Tabulating TREC data...")
    trec_df = dpp.trec_csv_from_dir(training_data_dir, trec_formatted_files, 'tabulated_unfiltered_trec.csv')
    trec_df = dpp.clean_text(trec_df, 'tabulated_cleaned_unfiltered_trec.csv')
    trec_df = dpp.persons_and_emotions(trec_df, 'tabulated_cleaned_emotionfiltered_trec.csv')
    
    def spam():
    ## Below is code to merge and create embeddings - currently unused:
    # # Merge the data
    # trec_df = dpp.merge_data(trec_df, training_rels_consenso_path)
    # print(f"Data merged successfully. Merged data size is: {trec_df.size}")

    # # Generate or load embeddings - embeddings for user generated posts. "embeddings.npy"
    # if os.path.exists('embeddings.npy'):
    #     print("Embeddings already exist. Loading...")
    #     trec_df['EMB'] = np.load('embeddings.npy', allow_pickle=True)
    # else:
    #     trec_df['EMB'] = trec_df['TEXT'].apply(generate_embeddings)
    #     np.save('embeddings.npy', trec_df['EMB'])
        pass

    # # Augmented answer sets from q1-q21. 
    aug_answers_df = dpp.process_augmented_data('augmented_answer_sets.txt', 'augmented_answer_sets.csv')


    '''
    COSINE SIMILARITY RANK CALCULATION
    The goal is to take the cosine similarity between the embeddings of the TREC data and the augmented data.
    In the augmented data, there are 21 questions, and each question has 4 severity levels for the answers.
    Therefore, a higher cosine similarity between text from a user generated post and 
    embeddings associated with symptoms should give that post a higher ranking for that particular question.
    From the user post data, the text comes from the TEXT column and the indicator for whether or not 
    the post has been associated with the question comes from the 'rel' column [0,1].
    '''
    print("Starting the cosine similarity rank calculation...")

    # Modify slightly - add a question_num argument, do for all 21 questions, and save to 21 dataframes
    # sorted on similarity for the particular question
    for i in range(1, 22):
        print(f"Calculating cosine similarity for question {i}...")
        cos_similarity_df = similarity_sum_over_col(trec_df, aug_answers_df, question_num=i)
        print(f"Saving cosine similarity for question {i}...")

    # ### This section is for evaluating the rank based on the 'rel' column. 
    # # Do the top 100 posts have relevance to the questions?
    # # Evaluate the relevance of the top 100 answers for each question
    # for i in range(1, 22):
    #     print(f"Evaluating relevance for question {i}...")
        
    #     # Load the cosine similarity dataframe for the question
    #     cos_similarity_df = pd.read_csv(f'cosine_similarity_{i}.csv')
        
    #     # Get the top 100 answers
    #     top_100_answers = cos_similarity_df.nlargest(100, 'cosine_similarity')
        
    #     # Check whether the answers are relevant to the question
    #     top_100_answers['is_relevant'] = top_100_answers['query'] == i
        
    #     # Calculate the precision of the top 100 answers
    #     precision = precision_score(top_100_answers['rel'], top_100_answers['is_relevant'])
        
    #     print(f"Precision for question {i}: {precision}")

    # Generate precision/accuracy tables for each question, and save the results to a csv


    
    # Generate plots for the precision/accuracy tables



    # Get overall performance stats for the model in general, and generate plots


    print("Program completed.")

if __name__ == '__main__':
    main()