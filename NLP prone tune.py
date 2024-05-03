import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Function to check if a word is a euphemism
def is_euphemism(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return len(set(synonyms)) > 1

# Function to find euphemisms in a sentence
def find_euphemisms(sentence):
    words = word_tokenize(sentence)
    euphemisms = [word for word in words if is_euphemism(word)]
    return euphemisms

# Read CSV file
def read_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found!")

# Main function
def main():
    # Replace 'your_file.csv' with the path to your CSV file
    file_path = "C:/Users/acer1/Downloads/EACL_english_dataset_1-24-24.csv"
    data = read_csv(file_path)
    if data is not None:
        for index, row in data.iterrows():
            text = row['sentence']  # Replace 'sentence' with the column containing the text
            
            # Basic preprocessing - convert text to lowercase
            text = text.lower()
            
            euphemisms = find_euphemisms(text)
            print(f"\nRow {index + 1} Text: {text}")
            print(f"Euphemisms found: {euphemisms}")
            
            # Manually inspect the euphemisms and update the dataset if necessary
            update_needed = input("Do you want to update the dataset for euphemisms? (y/n): ")
            if update_needed.lower() == 'y':
                updated_euphemisms = input("Enter the corrected euphemisms (comma-separated): ").split(',')
                data.at[index, 'euphemisms'] = ', '.join(updated_euphemisms)
                print("Dataset updated successfully!")
    
    # Save the updated dataset
    data.to_csv('updated_dataset.csv', index=False)

if __name__ == "__main__":
    main()
