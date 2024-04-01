import os
import pandas as pd
from data_processing_utils import (process_trec_dir, merge_dataframes, extract_polarity, 
                                   flag_self_referential, generate_embeddings)

def load_data(trec_csv1_path, training_data_dir, create_all_new):
    if not os.path.exists(trec_csv1_path) or create_all_new:
        trec_df = process_trec_dir(training_data_dir, sample=True, sample_size=100)
        trec_df.to_csv(trec_csv1_path, index=False)
    else:
        print('Reading in trec processed data...')
        trec_df = pd.read_csv(trec_csv1_path)
    return trec_df

def merge_data(trec_csv1merged_path, trec_df, training_qrels_majority_2, training_rels_consenso, create_all_new):
    if not os.path.exists(trec_csv1merged_path) or create_all_new:
        merged_data = merge_dataframes(trec_df, training_qrels_majority_2, training_rels_consenso)
        merged_data.to_csv(trec_csv1merged_path, index=False)
        print('Data merged')
    else:
        print("Reading in merged data...")
        merged_data = pd.read_csv(trec_csv1merged_path)
    return merged_data

def preprocess_data(merged_data):
    '''
    Preprocess the data by extracting the polarity of the text and flagging self-referential text.
    Add these factors as new columns.
    '''
    # First, drop rows that are NAN in the TEXT column
    merged_data = merged_data.dropna(subset=['TEXT'])

    merged_data['polarity'] = merged_data['TEXT'].apply(extract_polarity)
    # merged_data['self_reference'] = merged_data['TEXT'].apply(flag_self_referential)
    return merged_data

def main():
    # Set up the paths
    create_all_new = False
    top_level_dir = 't1_training_collection_2024/t1_training/TRAINING DATA (2023 COLLECTION)/eRisk2023_T1/'
    training_data_dir = 't1_training_collection_2024/t1_training/TRAINING DATA (2023 COLLECTION)/eRisk2023_T1/new_data'
    training_qrels_majority_2 = 't1_training_collection_2024/t1_training/TRAINING DATA (2023 COLLECTION)/eRisk2023_T1/g_qrels_majority_2.csv'
    training_rels_consenso = 't1_training_collection_2024/t1_training/TRAINING DATA (2023 COLLECTION)/eRisk2023_T1/g_rels_consenso.csv'
    trec_csv1_path = os.path.join(top_level_dir, 'preprocessed_data_v1.csv')
    trec_csv1merged_path = os.path.join(top_level_dir, 'merged_data_v1.csv')

    # Load and preprocess the data
    trec_df = load_data(trec_csv1_path, training_data_dir, create_all_new)
    merged_data = merge_data(trec_csv1merged_path, trec_df, training_qrels_majority_2, training_rels_consenso, create_all_new)
    merged_data = preprocess_data(merged_data)
    


    # Generate embeddings
    # embeddings = generate_embeddings(merged_data)

if __name__ == '__main__':
    main()