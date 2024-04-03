import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm



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

    # Generate or load embeddings - embeddings for user generated posts. "embeddings.npy"
    if os.path.exists('embeddings.npy'):
        print("Embeddings already exist. Loading...")
        trec_df['EMB'] = np.load('embeddings.npy', allow_pickle=True)
    else:
        trec_df['EMB'] = trec_df['TEXT'].apply(generate_embeddings)
        np.save('embeddings.npy', trec_df['EMB'])

    # Augmented answer sets from q1-q21. 
    # saved as "augmented_exploded.csv" and "augmented_exploded_embeddings.npy"
    aug_answers_df = process_augmented_data('augmented_answer_sets.txt', 'augmented_answers.csv',
                                            'augmented_exploded.csv', 'augmented_exploded_embeddings.npy')


    '''
    Vector embeddings have been generated for the TREC data and the augmented data.
    The goal is to take the cosine similarity between the embeddings of the TREC data and the augmented data.
    In the augmented data, there are 21 questions, and each question has 4 severity levels for the answers.
    Embeddings associated with severity levels 2, 3, and 4 are considered indicitative symptoms associated with the question.
    Therefore, a higher cosine similarity between text from a user generated post and 
    embeddings associated with symptoms will give that post a higher ranking for that particular question.
    From the user post data, the text comes from the TEXT column and the indicator for whether or not 
    the post has been associated with the question comes from the 'rel' column [0,1].
    We will split the data into training and validation, and evaluate the accuracy of our cosine similarity method.
    '''
    print("Starting the cosine similarity rank calculation...")

    # Split the data into training and validation
    train_df, val_df = train_test_split(trec_df, test_size=0.2, random_state=42)

    # Create a new column for the cosine similarity rank
    train_df['cosine_similarity_rank'] = 0


    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        # Get the trec_embedding for the current row
        trec_embedding = row['EMB']

        for question_num in range(1, 22):

            # add a column for each question
            train_df[f'cosine_similarity_rank_{question_num}'] = 0

            # Get the embeddings for the same question from the augmented data where severity is 2, 3, or 4
            aug_embeddings = aug_answers_df[(aug_answers_df['Question'] == question_num) 
                                            & (aug_answers_df['Severity'].isin([2, 3, 4]))]['EMB']

            # Calculate the cosine similarity for each augmented embedding and sum them
            similarity_sum = 0
            for aug_embedding in aug_embeddings:
                # Ensure the embeddings are numpy arrays
                if isinstance(trec_embedding, torch.Tensor):
                    trec_embedding = trec_embedding.cpu().numpy()
                if isinstance(aug_embedding, torch.Tensor):
                    aug_embedding = aug_embedding.cpu().numpy()

                # Calculate the cosine similarity
                similarity = cosine_similarity([trec_embedding], [aug_embedding])
                similarity_sum += similarity

            # Assign the sum of the similarities to the cosine_similarity_rank column
            train_df.at[index, f'cosine_similarity_rank_{question_num}'] = similarity_sum


    print("Program completed.")

if __name__ == '__main__':
    main()