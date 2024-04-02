import os
import pandas as pd
from data_processing_utils import *


def main():


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


    trec_df['EMB'] = trec_df['TEXT'].apply(generate_embeddings)

    print("Program completed.")

if __name__ == '__main__':
    main()