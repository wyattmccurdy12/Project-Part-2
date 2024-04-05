import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing_utils import *
from data_processing_utils import DataPreProcessor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score
from tqdm import tqdm
 ############
from sklearn.metrics import average_precision_score, ndcg_score
import numpy as np



# Set up a trec table with the 1000 top ranked sentences from each cosine similarity table
# five columns: item_number (a value from 1 to 21 identifying the Beck’s Depression Inventory item), the string “Q0” (for TREC format compliance), the sentence_id, position_in_ranking, score (cosine similarity), system_name
def create_trec_table(cosine_similarity_dfs, system_name):
    # Initialize an empty dataframe for the TREC table
    trec_table = pd.DataFrame(columns=['item_number', 'Q0', 'sentence_id', 'position_in_ranking', 'score', 'system_name'])

    # For each cosine similarity dataframe
    for i, df in enumerate(cosine_similarity_dfs):
        # Get the top 1000 sentences
        top_1000 = df.nlargest(1000, 'cosine_similarity')

        # Create a new dataframe with the required columns
        new_df = pd.DataFrame({
            'item_number': [i+1]*1000,  # item_number is the index of the dataframe in the list plus 1
            'Q0': ['Q0']*1000,  # Q0 is a constant string
            'sentence_id': top_1000.index,  # sentence_id is the index of the top 1000 sentences
            'position_in_ranking': range(1, 1001),  # position_in_ranking is a sequence from 1 to 1000
            'score': top_1000['cosine_similarity'],  # score is the cosine similarity of the top 1000 sentences
            'system_name': [system_name]*1000  # system_name is a constant string
        })

        # Append the new dataframe to the TREC table
        trec_table = pd.concat([trec_table, new_df])

    # Reset the index of the TREC table
    trec_table.reset_index(drop=True, inplace=True)

    return trec_table


def compute_metrics(trec_table):
    # Initialize a dictionary to store the metrics for each question
    metrics = {}

    # For each question
    for i in range(1, 22):
        # Get the rows for the question
        rows = trec_table[trec_table['item_number'] == i]

        # Compute the Average Precision (AP)
        ap = average_precision_score(rows['rel'], rows['score'])

        # Compute the R-Precision
        r_precision = rows['is_relevant'].sum() / len(rows)

        # Compute the Precision at 10
        precision_at_10 = rows['is_relevant'][:10].mean()

        # Compute the NDCG at 1000
        ndcg_at_1000 = ndcg_score(np.asarray([rows['is_relevant']]), np.asarray([rows['score']]))

        # Store the metrics for the question
        metrics[f'Question {i}'] = {
            'AP': ap,
            'R-Precision': r_precision,
            'Precision at 10': precision_at_10,
            'NDCG at 1000': ndcg_at_1000
        }

    return metrics












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
    dpp = DataPreProcessor(trec_formatted_files)

    print("Tabulating TREC data...")
    trec_df = dpp.trec_csv_from_dir(training_data_dir, trec_formatted_files, 'tabulated_unfiltered_trec.csv')
    trec_df = dpp.clean_text(trec_df, 'tabulated_cleaned_unfiltered_trec.csv')
    trec_df = dpp.persons_and_emotions(trec_df, 'tabulated_cleaned_emotionfiltered_trec.csv')
    
    # Useless things in a function for collapsing it
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
    '''
    print("Starting the cosine similarity rank calculation...")

    # Modify slightly - add a question_num argument, do for all 21 questions, and save to 21 dataframes
    # sorted on similarity for the particular question
    cosine_similarity_dfs = []
    for i in range(1, 22):
        print(f"Calculating cosine similarity for question {i}...")
        cos_similarity_df = similarity_sum_over_col(trec_df, aug_answers_df, question_num=i)
        print(f"Saving cosine similarity for question {i}...")
        cosine_similarity_dfs.append(cos_similarity_df)
    

    def spam():
        ### This section is for evaluating the rank based on the 'rel' column. 
        # Do the top 100 posts have relevance to the questions?
        # Evaluate the relevance of the top 100 answers for each question
        for i in range(1, 22):
            print(f"Evaluating relevance for question {i}...")
            
            # Load the cosine similarity dataframe for the question
            cos_similarity_df = pd.read_csv(f'cosine_similarity_{i}.csv')
            
            # Get the top 100 answers
            top_100_answers = cos_similarity_df.nlargest(100, 'cosine_similarity')
            
            # Check whether the answers are relevant to the question
            top_100_answers['is_relevant'] = top_100_answers['query'] == i
            
            # Calculate the precision of the top 100 answers
            precision = precision_score(top_100_answers['rel'], top_100_answers['is_relevant'])
            
            print(f"Precision for question {i}: {precision}")

    # Read in the rels table in order to evaluate the rankings - there are 2!
    training_rels_majority_path = os.path.join(training_data_dir, 'g_qrels_majority_2.csv')
    training_rels_consensus_path = os.path.join(training_data_dir, 'g_rels_consenso.csv')
    rels_majority_df = pd.read_csv(training_rels_majority_path)
    rels_consensus_df = pd.read_csv(training_rels_consensus_path)

    
    baseline_trec_table = create_trec_table(cosine_similarity_dfs, 'baseline')

    # Compute the metrics for the TREC table
    metrics = compute_metrics(baseline_trec_table)

    # Save metrics to a csv file
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv('metrics.csv')



    # Get overall performance stats for the model in general, and generate plots


    print("Program completed.")

if __name__ == '__main__':
    main()