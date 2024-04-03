import os
import numpy as np
import pandas as pd
from data_processing_utils import *

def main():
    """
    Main function that performs the following tasks:
    1. Load and preprocess the data from the training data directory.
    2. Merge the trec data with the consensus data.
    3. Clean the text in the merged data (remove duplicates, empty text, etc.)
    4. Create polarity column and a self reference flag column.
    5. Filter the data for negative and self referential sentences.
    6. Generate embeddings for the text data.
    7. Load the augmented data.
    8. Generate embeddings for the augmented data.
    """
    print("Starting the program...")

    # Define the top level directory
    top_level_dir = '/home/wyatt.mccurdy/Github/Project-Part-2/task1/'
    print(f"Top level directory is set to: {top_level_dir}")

    # Define the training data directory
    training_data_dir = os.path.join(top_level_dir, 'training/t1_training/TRAINING DATA (2023 COLLECTION)/')
    print(f"Training data directory is set to: {training_data_dir}")

    # Define the paths to the majority and consensus data
    training_qrels_majority_2 = os.path.join(training_data_dir, 'g_qrels_majority_2.csv')
    training_rels_consenso_path = os.path.join(training_data_dir, 'g_rels_consenso.csv')
    trec_formatted_files = os.path.join(training_data_dir, 'new_data/')

    # Load and preprocess the data
    print("Tabulating TREC data...")
    trec_df = trec_csv_from_dir(training_data_dir, trec_formatted_files)
    print("Data loaded successfully.")

    # Merge the data
    trec_df = merge_data(trec_df, training_rels_consenso_path)
    print(f"Data merged successfully. Merged data size is: {trec_df.size}")

    # Clean the text in the data
    trec_df = clean_text(trec_df)

    # Create a predominant polarity column, a self referential flag column, and filter the data
    trec_df = persons_and_emotions(trec_df)

    # Generate or load embeddings
    if os.path.exists('embeddings.npy'):
        print("Embeddings already exist. Loading...")
        trec_df['EMB'] = np.load('embeddings.npy', allow_pickle=True)
    else:
        trec_df['EMB'] = trec_df['TEXT'].apply(generate_embeddings)
        np.save('embeddings.npy', trec_df['EMB'])

    aug_answers_df = process_augmented_data('augmented_answer_sets.txt', 'augmented_answers.csv',
                                            'augmented_exploded.csv', 'augmented_exploded_embeddings.npy')

    print("Program completed.")

if __name__ == '__main__':
    main()