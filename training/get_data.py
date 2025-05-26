""" Download relevant data """

import nltk

def get_data():
    """Run data download step"""
    nltk.download('stopwords')

if __name__ == '__main__':
    get_data()
