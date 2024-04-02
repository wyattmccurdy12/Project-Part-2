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
    training_rels_consenso = 'g_rels_consenso.csv'
    trec_formatted_files = os.path.join(training_data_dir, 'new_data/')
    # print(os.listdir(trec_formatted_files))

    # Load and preprocess the data
    print("Tabulating TREC data...")
    trec_df = trec_csv_from_dir(training_data_dir, trec_formatted_files, 'tabulated_trec_data.csv', create_all_new)
    print(trec_df)
    print("Data loaded successfully.")

    # # print("Merging data...")
    # merged_data = merge_data('merged_data_v1.csv', trec_df, training_qrels_majority_2, training_rels_consenso, create_all_new)
    # # print("Data merged successfully.")

    # # Generate embeddings
    # # print("Generating embeddings...")
    # embeddings = generate_embeddings(merged_data)
    # # print("Embeddings generated successfully.")

    print("Program completed.")

if __name__ == '__main__':
    main()