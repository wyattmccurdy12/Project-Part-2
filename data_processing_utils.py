import os
import re
import numpy as np
import pandas as pd
import random 
from sklearn.metrics import average_precision_score, ndcg_score
from tqdm import tqdm
import nltk
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool


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
            df = self.process_trec_file(directory, filename)
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
            trec_df = self.process_trec_dir(trec_data_path, sample=False)
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
            merged_data = self.merge_dataframes(trec_df,  training_rels_consenso)
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

## LANGUAGE PROCESSING FUNCTIONS ##
## Extract polarity, flag self-referential sentences, and filter positive and neutral sentences
class LanguageProcessor:
    """
    Language processing utilities.

    Attributes:
        sia (SentimentIntensityAnalyzer): An instance of the VADER sentiment analysis tool.

    Methods:
        extract_polarity: Extracts the polarity of a text using the VADER sentiment analysis tool.
        flag_self_referential: Flags self-referential sentences in a text.
        filter_positive_and_neutral_sents: Filters out sentences that are positive, neutral, or non-negative.
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def extract_polarity(self, text):
        """
        Extract the polarity of a text using the VADER sentiment analysis tool.

        Args:
            text (str): The text to analyze.
        
        Returns:
            str: The key ('neg', 'neu', 'pos') corresponding to the maximum polarity score.
        """
        polarity = self.sia.polarity_scores(text)
        pp = polarity.pop('compound') # remove compound score. remains unused
        keys, values = zip(*polarity.items())
        max_key = keys[values.index(max(values))]

        return max_key

    def flag_self_referential(self, text):
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

    def filter_positive_and_neutral_sents(self, sentences):
        """
        Filter out sentences that are positive, neutral, or non-negative.

        Args:
            sentences (list): A list of sentences to filter.
        
        Returns:
            list: A list of sentences that are negative.
        """
        negative_sentences = []
        for sentence in sentences:
            sentiment = self.extract_polarity(sentence)
            if sentiment == 'neg':
                negative_sentences.append(sentence)
        return negative_sentences
## END LANGUAGE PROCESSING FUNCTIONS ##

## VECTOR EMBEDDING FUNCTIONS ##
## Generate vector embeddings and process them for cosine similarity calculations
class EmbeddingProcessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def calculate_similarity_sum(self, input_text, aug_answers_df, df_column):
        """
        Calculates the sum of cosine similarity scores between the input text and each answer text in an input dataframe.

        Args:
            input_text (str): The input text to compare against.
            aug_answers_df (pandas.DataFrame): The input dataframe containing answer texts.
            df_column (str): The column name in the dataframe that contains the answer texts.

        Returns:
            float: The sum of cosine similarity scores.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        inputs_1 = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs_1 = self.model(**inputs_1)
        cs_sum = 0
        for answer_text in aug_answers_df[df_column]:
            inputs_2 = self.tokenizer(answer_text, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs_2 = self.model(**inputs_2)
            cs = cosine_similarity(outputs_1.last_hidden_state.mean(dim=1).detach().cpu().numpy(), 
                                   outputs_2.last_hidden_state.mean(dim=1).detach().cpu().numpy())
            cs_sum += cs[0][0]
        return cs_sum


    def calculate_similarity_for_row(self, row, corresponding_answer):
        cs_sum = self.calculate_similarity_sum(row['TEXT'], corresponding_answer, 'Text')
        return cs_sum

    def similarity_sum_over_col(self, persons_and_emotions_df, augmented_exploded_df, question_num):
        """
        Calculates the similarity sum over a specific bdi query in the persons_and_emotions_df DataFrame.
        This takes a long time due to n cosine similarity calculations for each row in the dataframe for the specific question.

        Args:
            persons_and_emotions_df (DataFrame): The DataFrame containing persons and emotions data.
            augmented_exploded_df (DataFrame): The DataFrame containing augmented and exploded data.
            question_num (int): The question number for which the similarity sum is calculated.

        Returns:
            DataFrame: The updated persons_and_emotions_df DataFrame with the similarity sum column added.

        Raises:
            None
        """
        save_name = f'cosine_similarity_q{question_num}'
        if os.path.exists(save_name):
            persons_and_emotions_df = pd.read_csv(save_name)
        else:
            persons_and_emotions_df[f'SIM_{question_num}'] = ''
            corresponding_answer = augmented_exploded_df[(augmented_exploded_df['Question'] == question_num)]
            with Pool() as p:
                cs_sums = p.starmap(self.calculate_similarity_for_row, [(row, corresponding_answer) for _, row in persons_and_emotions_df.iterrows()])
            persons_and_emotions_df[f'SIM_{question_num}'] = cs_sums
            persons_and_emotions_df = persons_and_emotions_df.sort_values(by=f'SIM_{question_num}', ascending=False)
            persons_and_emotions_df.to_csv(save_name, index=False)
            print(f"Data saved to {save_name}.")
        return persons_and_emotions_df

# POST PROCESSING FUNCTIONS - TREC TABLE AND METRICS - Accuracy etc.
class PostProcessor:
    @staticmethod
    def create_trec_table(cosine_similarity_dfs, system_name, ground_truth_df):
        """
        Create a TREC table based on cosine similarity data, for each BDI question.

        Parameters:
        cosine_similarity_dfs (list): A list of pandas DataFrames containing cosine similarity data - outputs.
        system_name (str): The name of the system.
        ground_truth_df (pandas DataFrame): The ground truth DataFrame.

        Returns:
        pandas DataFrame: The TREC table.
        """
        trec_table = pd.DataFrame(columns=['item_number', 'Q0', 'sentence_id', 'position_in_ranking', 'score', 'system_name', 'rel'])

        for i, df in enumerate(cosine_similarity_dfs):
            top_1000 = df.head(1000)
            top_1000['item_number'] = i+1
            sim_col_name = f'SIM_{i+1}'
            top_1000 = top_1000.rename(columns={
                'docid': 'sentence_id', 
                sim_col_name: 'cosine_similarity'})

            top_1000['q0'] = ''
            top_1000['system_name'] = system_name
            top_1000 = top_1000[['item_number', 'q0', 'sentence_id', 'cosine_similarity', 'system_name']]

            new_df = top_1000.merge(ground_truth_df, left_on=['item_number', 'sentence_id'], right_on=['query', 'sentence_id'], how='left')

            trec_table = pd.concat([trec_table, new_df])

        trec_table.reset_index(drop=True, inplace=True)

        return trec_table

    @staticmethod
    def compute_metrics(trec_table):
        """
        Compute evaluation metrics for each question in the TREC table.

        Parameters:
        trec_table (pd.DataFrame): DataFrame containing the top 1000 sentences for each query, 
                                   along with their cosine similarity scores, system name, and relevance data.

        Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a question and the columns are the computed metrics.
        """
        metrics = []

        for i in range(1, 22):
            rows = trec_table[trec_table['item_number'] == i]

            ap = average_precision_score(rows['rel'], rows['score'])

            r_precision = rows['is_relevant'].sum() / len(rows)

            precision_at_10 = rows['is_relevant'][:10].mean()

            ndcg_at_1000 = ndcg_score(np.asarray([rows['is_relevant']]), np.asarray([rows['score']]))

            metrics.append({
                'Question': i,
                'AP': ap,
                'R-Precision': r_precision,
                'Precision at 10': precision_at_10,
                'NDCG at 1000': ndcg_at_1000
            })

        return pd.DataFrame(metrics)