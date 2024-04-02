import os
import re
import pandas as pd
import random 
from tqdm import tqdm
import nltk
import yake
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
import torch



## PREPROCESSING FUNCTIONS ##
def process_trec_file(directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as f:
        content = f.read()

    df = pd.DataFrame(columns=['docid', 'PRE', 'TEXT', 'POST'])

    doctags = re.findall('<DOC>.*?</DOC>', content, re.DOTALL)
    for doctag in doctags:
        docno = re.findall('<DOCNO>(.*?)</DOCNO>', doctag)[0]
        pre = re.findall('<PRE>(.*?)</PRE>', doctag)
        pre = pre[0] if pre else ''
        text = re.findall('<TEXT>(.*?)</TEXT>', doctag)
        text = text[0] if text else ''
        post = re.findall('<POST>(.*?)</POST>', doctag)
        post = post[0] if post else ''
        new_row = pd.DataFrame([[docno, pre, text, post]], columns=['docid', 'PRE', 'TEXT', 'POST'])
        df = pd.concat([df, new_row], ignore_index=True)

    df = df.dropna(subset=['docid'])
    df = df[['docid', 'PRE', 'TEXT', 'POST']]

    return df


def process_trec_dir(directory, sample=False, sample_size=100):
    """
    Process all TREC files in a directory and return a concatenated DataFrame.

    Args:
        directory (str): The directory path where the TREC files are located.
        sample (bool, optional): Whether to sample files randomly. Defaults to True.
        sample_size (int, optional): The number of files to sample. Defaults to 100.

    Returns:
        pandas.DataFrame: A concatenated DataFrame containing the processed data from all TREC files.
    """
    filelist = os.listdir(directory)
    if sample:
        filelist = random.sample(filelist, sample_size)
    df_list = []
    for filename in tqdm(filelist, desc='Processing TREC files'):
        df = process_trec_file(directory, filename)
        df_list.append(df)
    result_df = pd.concat(df_list, ignore_index=True)
    return result_df


def merge_dataframes(trec_df, training_qrels_majority_2, training_rels_consenso):
    # Load the ancillary data
    qrels = pd.read_csv(training_qrels_majority_2)
    rels = pd.read_csv(training_rels_consenso)

    # Print the size of each dataframe
    print(f'Size of qrels: {qrels.shape[0]}')
    print(f'Size of rels: {rels.shape[0]}')
    print(f'Size of trec_df: {trec_df.shape[0]}')

    # Merge qrels and rels on docid
    merged_data = pd.merge(qrels, rels, on='docid', how='inner')
    print(f'Size of merged_data (qrels and rels): {merged_data.shape[0]}')

    # Merge trec_df and merged_data on docid
    merged_data = pd.merge(trec_df, merged_data, on='docid', how='inner')
    print(f'Size of merged_data (trec_df and merged_data): {merged_data.shape[0]}')

    return merged_data

## END PREPROCESSING FUNCTIONS ##
############################################################################################

## Functions pasted from main - modularization functions ##

def trec_csv_from_dir(training_data_dir, trec_folder_name, output_csv_path='tabulated_trec_data.csv'):
    """
    Load and preprocess TREC data from the directory and save it to a CSV file. (if not exists already)

    Parameters:
    - trec_csv1_path (str): The path to the TREC CSV file. (output)
    - training_data_dir (str): The directory containing the TREC formatted data.
    - create_all_new (bool): If True, the training data will be processed again and saved as a new CSV file. (will replace)

    Returns:
    - trec_df (pandas.DataFrame): The loaded or processed TREC data as a DataFrame.
    """
    if os.path.exists(output_csv_path):
        print('Reading in trec processed data...')
        trec_df = pd.read_csv(output_csv_path)
        return trec_df
    else:
        print('Tabulating TREC data...')
        trec_data_path = os.path.join(training_data_dir, trec_folder_name)
        trec_df = process_trec_dir(trec_data_path, sample=False)
        trec_df.to_csv(output_csv_path, index=False)
        return trec_df


def merge_data(trec_df, training_qrels_majority_2, training_rels_consenso, out_merged_csv_path='merged_data.csv', create_all_new=False):
    """
    Merge the given dataframes and save the merged data to a CSV file. (if not exists already)

    Args:
        trec_csv1merged_path (str): The file path to save the merged data CSV file.
        trec_df (pandas.DataFrame): The dataframe to be merged.
        training_qrels_majority_2 (pandas.DataFrame): The dataframe containing training qrels majority 2 data.
        training_rels_consenso (pandas.DataFrame): The dataframe containing training rels consenso data.
        create_all_new (bool): If True, create a new merged data file even if it already exists.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """
    if not os.path.exists(out_merged_csv_path) or create_all_new:
        merged_data = merge_dataframes(trec_df, training_qrels_majority_2, training_rels_consenso)
        merged_data.to_csv(out_merged_csv_path, index=False)
        print('Data merged')
    else:
        print("Reading in merged data...")
        merged_data = pd.read_csv(out_merged_csv_path)
    return merged_data

def preprocess_data(merged_data):
    '''
    Preprocess the data by extracting the polarity of the text and flagging self-referential text.
    Add these factors as new columns.

    Parameters:
    - merged_data: pandas DataFrame
        The input data containing the text column.

    Returns:
    - preprocessed_data: pandas DataFrame
        The preprocessed data with additional columns for polarity and self-reference flags.
    '''
    # First, drop rows that are NAN in the TEXT column
    merged_data = merged_data.dropna(subset=['TEXT'])

    merged_data['polarity'] = merged_data['TEXT'].apply(extract_polarity)
    # merged_data['self_reference'] = merged_data['TEXT'].apply(flag_self_referential)
    return merged_data

## END Functions pasted from main - modularization functions ##

## LANGUAGE PROCESSING FUNCTIONS ##

def extract_polarity(text):
    """
    Extract the polarity of a text using the VADER sentiment analysis tool.

    Args:
        text (str): The text to analyze.
    
    Returns:
        dict: A dictionary containing the polarity scores. Keys: 'neg', 'neu', 'pos', 'compound'. 
                                                    negative, neutral, positive, and compound scores.
    """
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(text)

    # Get the maximum polarity score, return it and its corresponding key in a tuple
    max_score = 0
    for key, value in polarity.items():
        if value > max_score:
            max_score = value
            max_score_key = key

    return max_score_key

def flag_self_referential(text):
    """
    Flag self-referential sentences in a text.

    Args:
        text (str): The text to analyze.
    
    Returns:
        list: A list of tuples, where each tuple contains a sentence and a flag indicating whether the sentence is self-referential.
    """
    sentences = sent_tokenize(text)
    # Tokenize each sentence
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    # Assign part of speech tags to the tokens
    pos_tagged_sentences = [nltk.pos_tag(tokens) for tokens in tokenized_sentences]

    # Flag self-referential sentences
    self_referential_sentences = []
    for sentence, pos_tags in zip(sentences, pos_tagged_sentences):
        # Check if the sentence is self-referential
        is_self_referential = any(tag == 'PRP' for word, tag in pos_tags)
        self_referential_sentences.append((sentence, int(is_self_referential)))

    return self_referential_sentences

def filter_positive_and_neutral_sents(sentences):
    """
    Filter out sentences that are positive, neutral, or non-negative.

    Args:
        sentences (list): A list of sentences to filter.
    
    Returns:
        list: A list of sentences that are negative.
    """
    negative_sentences = []
    for sentence in sentences:
        sentiment = extract_polarity(sentence)
        if sentiment == 'neg':
            negative_sentences.append(sentence)
    return negative_sentences

## END LANGUAGE PROCESSING FUNCTIONS ##

## VECTOR EMBEDDING FUNCTIONS ##
## Generate vector embeddings
def generate_embeddings(text):
    """
    Generate vector embeddings for a text using the all-MiniLM-L6-v2 pretrained model.

    Args:
        text (str): The text to generate embeddings for.
    
    Returns:
        torch.Tensor: The embeddings for the input text.
    """
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Generate the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Take the mean of the token embeddings as the sentence embedding

    return embeddings

def create_embeddings_for_sentences(sentences):
    """
    Generate embeddings for a list of sentences.

    Args:
        sentences (list): A list of sentences to generate embeddings for.
    
    Returns:
        torch.Tensor: The embeddings for the input sentences.
    """
    embeddings_list = []
    for sentence in sentences:
        embeddings = generate_embeddings(sentence)
        embeddings_list.append(embeddings)
    return torch.cat(embeddings_list, dim=0)

## END VECTOR EMBEDDING FUNCTIONS ##

# TODO test this function
def load_augmented_answers(augmented_answers_file):
    """
    Load augmented answers from a text file and process them into a dictionary.

    Args:
        augmented_answers_file (str): The path to the text file containing the augmented answers.
    
    Returns:
        dict: The augmented answers as a dictionary.
    """
    with open(augmented_answers_file, 'r') as file:
        lines = file.readlines()

    # Initialize the dictionary
    augmented_answers = {}

    # Process each line
    for line in lines:
        # Split the line into segments
        segments = line.split('Equivalent answers for item ')[1:]
        for segment in segments:
            # Extract the question number and the answers
            question_number, answers_segment = segment.split('\n', 1)
            answers = answers_segment.split('} {')
            answers[0] = answers[0].lstrip('{ ')
            answers[-1] = answers[-1].rstrip(' }')

            # Process the answers into a subdictionary
            subdictionary = {}
            for answer in answers:
                number, text = answer.split('. ', 1)
                subdictionary[number] = text.split(', ')

            # Add the subdictionary to the main dictionary
            augmented_answers[f'Question {question_number}'] = subdictionary

    return augmented_answers