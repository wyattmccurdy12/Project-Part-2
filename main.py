import os
import numpy as np
import pandas as pd
from data_processing_utils import *


def main():
    """
    1. Load and preprocess the data from the training data directory.
    2. Merge the trec data with the consensus data.
    3. Clean the text in the merged data (remove duplicates, empty text, etc.)
    4. Create polarity column and a self reference flag column.
    5. Filter the data for negative and self referential sentences.
    6. Generate embeddings for the text data.
    7. Load the augmented data.
    8. Generate embeddings for the augmented data.

    The function does not take any arguments and does not return any values. 
    It prints messages to the console to indicate the progress of the tasks.
    """
    print("Starting the program...")

    # Set up the paths
    create_all_new = True
    top_level_dir = '/home/wyatt.mccurdy/Github/Project-Part-2/task1/'
    print(f"Top level directory is set to: {top_level_dir}")

    # Set up training data directory
    training_data_dir = 'training/t1_training/TRAINING DATA (2023 COLLECTION)/'
    training_data_dir = os.path.join(top_level_dir, training_data_dir)
    print(f"Training data directory is set to: {training_data_dir}")

    training_qrels_majority_2 = 'g_qrels_majority_2.csv'
    training_qrels_majority_2 = os.path.join(training_data_dir, training_qrels_majority_2)
    training_rels_consenso_name = 'g_rels_consenso.csv'
    training_rels_consenso_path = os.path.join(training_data_dir, training_rels_consenso_name)
    trec_formatted_files = os.path.join(training_data_dir, 'new_data/')
    # print(os.listdir(trec_formatted_files))

    # Load and preprocess the data
    print("Tabulating TREC data...")
    trec_df = trec_csv_from_dir(training_data_dir, trec_formatted_files)
    print("Data loaded successfully.")

    # print("Merging data...")
    trec_df = merge_data(trec_df, training_rels_consenso_path)
    print(f"Data merged successfully. Merged data size is: {trec_df.size}")

    # Remove data with no text
    trec_df = clean_text(trec_df)

    # Create a predominant polarity column, then a self referential flag column, then filter the data for negative and self referential sentences
    trec_df = persons_and_emotions(trec_df)

    # Generate embeddings
    if os.path.exists('embeddings.npy'):
        print("Embeddings already exist. Loading...")
        trec_df['EMB'] = np.load('embeddings.npy', allow_pickle=True)
    else:
        trec_df['EMB'] = trec_df['TEXT'].apply(generate_embeddings)
        np.save('embeddings.npy', trec_df['EMB'])

    ### Augmented Data ###
    # Read in relevant answers to the 21 questions
    aug_answers_df = generate_answers_df(in_lines_file='augmented_answer_sets.txt', out_file_path='augmented_answers.csv')
    print("Augmented answers loaded successfully.")

    # Split the answers into individual sentences
    aug_answers_df['Text'] = aug_answers_df['Text'].str.split(',')
    aug_answers_df = aug_answers_df.explode('Text')
    
    # Generate embeddings for the answers
    aug_answers_df['EMB'] = aug_answers_df['Text'].apply(generate_embeddings)
    ### End Augmented Data ###



    print("Program completed.")

if __name__ == '__main__':
    main()