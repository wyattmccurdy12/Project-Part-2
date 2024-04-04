import os
import re
import numpy as np
import pandas as pd
import random 
from tqdm import tqdm
import nltk
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

##########################################
## PREPROCESSING FUNCTIONS ## 
##############
######

class DataPreProcessor:
    """
    Preprocess TREC formatted files as well as rels dataframes.

    Args:
        trec_dir (str): The directory path where the TREC files are located.

    Attributes:
        trec_dir (str): The directory path where the TREC files are located.
    """
    def __init__(self, trec_dir):
        self.trec_dir = trec_dir

    def process_trec_file(self, directory, filename):
        """
        Process a TREC file and return a pandas DataFrame. A building block for process_trec_dir.

        Parameters:
        directory (str): The directory where the TREC file is located.
        filename (str): The name of the TREC file.

        Returns:
        pandas.DataFrame: A DataFrame containing the processed data from the TREC file.
        """
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

    def process_trec_dir(self, directory, sample=False, sample_size=100):
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

    def trec_csv_from_dir(self, training_data_dir, trec_folder_name, output_csv_path='tabulated_trec_data.csv'):
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

    def merge_dataframes(self, trec_df, training_rels_consenso):
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

    def merge_data(self, trec_df, training_rels_consenso_path, out_merged_csv_path='merged_data.csv'):
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

    def clean_text(self, df, outname='merged_clean_data.csv'):
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
            print(f"Data size after removing rows with no text: {df.shape}")
            
            df = df[df['TEXT'] != '']
            print(f"Data size after removing rows with empty text: {df.shape}")
            
            print("Removing duplicate rows...")
            df = df.drop_duplicates(subset=['TEXT'])
            print(f"Data size after removing duplicate rows: {df.shape}")
            
            df.to_csv(outname, index=False)
            print(f"Data cleaned and saved to {outname}.")
        return df

    def persons_and_emotions(self, df, outname='persons_and_emotions.csv'):
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

    def generate_answers_df(self, in_lines_file='augmented_answer_sets.txt'):
        # if os.path.exists(out_file_path):
        #     return pd.read_csv(out_file_path)

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

    def process_augmented_data(self, in_lines_file, exploded_df_path):
        """
        This function loads the augmented answer sets from BDI, a
        and processes them, outputing a dataframe and csv.

        Parameters:
        in_lines_file (str): The path to the input file containing the augmented data.
        exploded_df_path (str): The path to the saved exploded dataframe.
        embeddings_path (str): The path to the saved embeddings.

        Returns:
        DataFrame: A pandas DataFrame containing the processed augmented data.
        """
        if os.path.exists(exploded_df_path):
            print("Loading exploded dataframe from disk...")
            aug_answers_df = pd.read_csv(exploded_df_path)
        else:
            print("Generating exploded augmented answers dataframe...")
            # Load the augmented data
            aug_answers_df = self.generate_answers_df(in_lines_file)

            # Split the answers into individual sentences
            aug_answers_df['Text'] = aug_answers_df['Text'].str.split(',')
            aug_answers_df = aug_answers_df.explode('Text')

            # Save the exploded dataframe and embeddings
            aug_answers_df.to_csv(exploded_df_path, index=False)
            print(f"Exploded dataframe saved to {exploded_df_path}.")
        print("Augmented answer sets processed.\n")
        return aug_answers_df


## Functions pasted from main - modularization functions ##



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

## COSINE SIMILARITY FUNCTIONS ##
# Maybe not used...
def calculate_similarity(sentence_1, sentence_2, tokenizer, model):
    """
    This function calculates the cosine similarity between the embeddings of two sentences.

    Parameters:
    sentence_1 (str): The first sentence.
    sentence_2 (str): The second sentence.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
    model (transformers.PreTrainedModel): The model to use.

    Returns:
    cs (float): The cosine similarity between the embeddings of the two sentences.
    """
    # Tokenize and get embeddings for the sentences
    inputs_1 = tokenizer(sentence_1, return_tensors='pt', padding=True, truncation=True)
    inputs_2 = tokenizer(sentence_2, return_tensors='pt', padding=True, truncation=True)
    outputs_1 = model(**inputs_1)
    outputs_2 = model(**inputs_2)

    # Calculate cosine similarity between the embeddings
    cs = cosine_similarity(outputs_1.last_hidden_state.mean(dim=1).detach().numpy(), 
                           outputs_2.last_hidden_state.mean(dim=1).detach().numpy())

    return cs[0][0]


# Create a function to calculate the sum of cosine similarities between an input text and a dataframe column of answer texts
def calculate_similarity_sum(input_text, input_df, df_column, tokenizer, model):
    """
    This function calculates the sum of cosine similarities between the input text and a dataframe column of answer texts.

    Parameters:
    input_text (str): The input text.
    input_df (pandas.DataFrame): The dataframe containing the answer texts.
    df_column (str): The column in the dataframe containing the answer texts.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
    model (transformers.PreTrainedModel): The model to use.

    Returns:
    cs_sum (float): The sum of cosine similarities between the input text and the answer texts.
    """
    # Tokenize and get embeddings for the input text
    inputs_1 = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    outputs_1 = model(**inputs_1)

    # Calculate cosine similarity between the input text and each answer text
    cs_sum = 0
    for answer_text in input_df[df_column]:
        inputs_2 = tokenizer(answer_text, return_tensors='pt', padding=True, truncation=True)
        outputs_2 = model(**inputs_2)
        cs = cosine_similarity(outputs_1.last_hidden_state.mean(dim=1).detach().numpy(), 
                               outputs_2.last_hidden_state.mean(dim=1).detach().numpy())
        cs_sum += cs[0][0]

    return cs_sum


def similarity_sum_over_col(persons_and_emotions_df, augmented_exploded_df, question_num):
    """
    This function reads the persons_and_emotions dataframe and creates answer columns for each of the 21 questions.
    Then for each question, it finds the corresponding answer in the augmented_exploded dataframe and gets the cosine similarity sum.

    Parameters:
    persons_and_emotions_file (str): The path to the persons_and_emotions CSV file.
    augmented_exploded_file (str): The path to the augmented_exploded CSV file.

    Returns:
    persons_and_emotions_df (pandas.DataFrame): The processed dataframe.
    """

    # Specify the name of the CSV file
    save_name = f'cosine_similarity_q{question_num}'

    # Check if the CSV file exists
    if os.path.exists(save_name):
        # If the file exists, load the dataframe from it
        persons_and_emotions_df = pd.read_csv(save_name)
    else:
        # If the file doesn't exist, perform the calculations

        # Load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create answer column for the question and its similarity score
        persons_and_emotions_df[f'SIM_{question_num}'] = ''

        # For each question, find the corresponding answer in the augmented_exploded dataframe and get the cosine similarity sum

        for index, row in tqdm(persons_and_emotions_df.iterrows(), total=persons_and_emotions_df.shape[0]):
            corresponding_answer = augmented_exploded_df[(augmented_exploded_df['Question'] == question_num)]
            cs_sum = calculate_similarity_sum(row['TEXT'], corresponding_answer, 'Text', tokenizer, model)
            persons_and_emotions_df.at[index, f'SIM_{question_num}'] = cs_sum

        # Sort the dataframe by similarity
        persons_and_emotions_df = persons_and_emotions_df.sort_values(by=f'SIM_{question_num}', ascending=False)

        # Save the resulting dataframe to a CSV file
        persons_and_emotions_df.to_csv(save_name, index=False)
        print(f"Data saved to {save_name}.")

    return persons_and_emotions_df

# Updating this function. See above
def calculate_cosine_similarity(post_text_df, aug_answers_df, save_name):
    """
    This function calculates the cosine similarity between the embeddings of each row in the incoming posts and the embeddings
    of each answer in the augmented data where severity is 2, 3, or 4. The results are stored in new columns in
    the post_text_df.

    If a save_name is provided, the function will attempt to load the dataframe from a CSV file with that name. If the file
    does not exist, it will perform the calculations and save the resulting dataframe to a CSV file with the provided name.

    Parameters:
    post_text_df (pandas.DataFrame): The dataframe containing the post texts. It should have a column 'EMB' for embeddings.
    aug_answers_df (pandas.DataFrame): The dataframe containing the augmented data. It should have a column 'Question' for question numbers,
                                       a column 'Severity' for severity levels, and a column 'EMB' for embeddings.
    save_name (str): The name of the CSV file to save to or load from.

    Returns:
    post_text_df (pandas.DataFrame): The input dataframe with added cosine similarity rank columns.
    """
    if os.path.exists(save_name):
        post_text_df = pd.read_csv(save_name)
    else:
        for index, row in tqdm(post_text_df.iterrows(), total=post_text_df.shape[0]):
            trec_embedding = row['EMB']

            for question_num in range(1, 22):
                post_text_df[f'cosine_similarity_rank_{question_num}'] = 0

                aug_embeddings = aug_answers_df[(aug_answers_df['Question'] == question_num) 
                                                & (aug_answers_df['Severity'].isin([2, 3, 4]))]['EMB']

                similarity_sum = 0
                for aug_embedding in aug_embeddings:
                    if isinstance(trec_embedding, torch.Tensor):
                        trec_embedding = trec_embedding.cpu().numpy()
                    if isinstance(aug_embedding, torch.Tensor):
                        aug_embedding = aug_embedding.cpu().numpy()

                    similarity = cosine_similarity(trec_embedding.reshape(1, -1), aug_embedding.reshape(1, -1))
                    similarity_sum += similarity

                post_text_df.at[index, f'cosine_similarity_rank_{question_num}'] = similarity_sum

        post_text_df.to_csv(save_name, index=False)

    return post_text_df



## END COSINE SIMILARITY FUNCTIONS ##

## LET'S CLEAN UP AND PUT UNUSED FUNCTIONS IN THIS LAST PART
# { # UNUSED FUNCTIONS ## 












## } END UNUSED FUNCTIONS ##