import os
import re
import pandas as pd
import random 
from tqdm import tqdm
import nltk
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
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


def merge_dataframes(trec_df, training_rels_consenso):
    # Load the ancillary data
    # qrels = pd.read_csv(training_qrels_majority_2)
    # rels = pd.read_csv(training_rels_consenso)

    # Print the size of each dataframe
    # print(f'Size of qrels: {qrels.shape[0]}')
    print(f'Size of rels: {training_rels_consenso.shape[0]}')
    print(f'Size of trec_df: {trec_df.shape[0]}')

    # Merge qrels and rels on docid
    # merged_data = pd.merge(qrels, rels, on='docid', how='inner')
    # print(f'Size of merged_data (qrels and rels): {merged_data.shape[0]}')

    # Merge trec_df and merged_data on docid
    merged_data = pd.merge(trec_df, training_rels_consenso, on='docid', how='inner')
    print(f'Size of merged_data (trec_df and merged_data): {merged_data.shape[0]}')

    return merged_data


def clean_text(df, outname='merged_clean_data.csv'):
    """
    Remove rows with no text or with empty strings in the 'TEXT' column, and remove duplicate rows based on the 'TEXT' column.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    if os.path.exists(outname):
        print(f"Loading data from {outname}...")
        df = pd.read_csv(outname)
    else:
        print("Removing data with no text...")
        df = df.dropna(subset=['TEXT'])
        df = df[df['TEXT'] != '']
        print("Removing duplicate rows...")
        df = df.drop_duplicates(subset=['TEXT'])
        df.to_csv(outname, index=False)
        print(f"Data cleaned and saved to {outname}.")
    return df


def persons_and_emotions(df, outname='persons_and_emotions.csv'):
    """
    Create a predominant polarity column, a self referential flag column, and filter the data for negative and self referential sentences.

    Args:
        df (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    if os.path.exists(outname):
        print(f"Loading data from {outname}...")
        df = pd.read_csv(outname)
    else:
        print("Creating predominant polarity column...")
        df['polarity'] = df['TEXT'].apply(extract_polarity)
        print("Creating self reference flags...")
        df['self_ref'] = df['TEXT'].apply(flag_self_referential)
        print("Filtering to only include negative and self referential posts...")
        df = df[(df['polarity'] == 'neg' ) & (df['self_ref'] == 1)]

        # Save filter flagged data to 'persons_and_emotions.csv'
        df.to_csv(outname, index=False)
        print(f"Data processed and saved to {outname}.")
    return df


def generate_answers_df(in_lines_file='augmented_answer_sets.txt', out_file_path='aug_answers.csv'):
    if os.path.exists(out_file_path):
        return pd.read_csv(out_file_path)

    questions = {
        i: {j: [] for j in range(1, 5)}
        for i in range(1, 22)
    }

    with open(in_lines_file, 'r') as f:
        lines = f.readlines()
    
    question_number = 0
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            question_number = int(line)
            severity = 1
        else:
            questions[question_number][severity].append(line)
            severity += 1

    df_list = []
    for question_number in questions:
        for severity in questions[question_number]:
            for text in questions[question_number][severity]:
                df_list.append(pd.DataFrame({'Question': [question_number], 'Severity': [severity], 'Text': [text]}))
    df = pd.concat(df_list, ignore_index=True)

    return df


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


def merge_data(trec_df, training_rels_consenso_path, out_merged_csv_path='merged_data.csv'):
    """
    Merge the given dataframes and save the merged data to a CSV file. (if not exists already)

    Args:
        trec_csv1merged_path (str): The file path to save the merged data CSV file.
        trec_df (pandas.DataFrame): The dataframe to be merged.
        training_qrels_majority_2 (pandas.DataFrame): The dataframe containing training qrels majority 2 data.
        training_rels_consenso (pandas.DataFrame): The dataframe containing training rels consenso data.

    Returns:
        pandas.DataFrame: The merged dataframe.
    """
    
    if not os.path.exists(out_merged_csv_path):
        training_rels_consenso = pd.read_csv(training_rels_consenso_path)
        merged_data = merge_dataframes(trec_df,  training_rels_consenso)
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
    pp = polarity.pop('compound') # remove compound score. remains unused
    keys, values = zip(*polarity.items())
    max_key = keys[values.index(max(values))]

    return max_key

def flag_self_referential(text):
    """
    Flag self-referential sentences in a text.

    Args:
        text (str): The text to analyze.
    
    Returns:
        int: 1 if the text is self-referential, 0 otherwise.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Check if the text is self-referential
    is_self_referential = any(tag == 'PRP' for word, tag in pos_tags)

    return int(is_self_referential)

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

