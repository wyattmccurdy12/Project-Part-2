import xml.etree.ElementTree as ET
import re
import pandas as pd

# XML Processing Functions
def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def replace_urls_with_marker(root):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    for element in root.iter():
        if element.text:
            element.text = url_pattern.sub('[URL]', element.text)
    return root

def preprocess_data(file_path):
    root = read_xml(file_path)
    root = replace_urls_with_marker(root)
    return root

# Data Conversion Function
def xml_to_df(root):
    data = []
    for element in root.iter():
        data.append(element.attrib)
    df = pd.DataFrame(data)
    return df

# Data Writing Function
def write_df_to_csv(df, file_name):
    df.to_csv(file_name, index=False)

# Main Execution
root = preprocess_data('path_to_your_file.xml')
df = xml_to_df(root)
write_df_to_csv(df, 'output.csv')