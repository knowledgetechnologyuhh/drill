import re
import pandas as pd
from torch.utils import data


def preprocess(text):
    # lower-case characters
    text = text.lower()
    # Change 't to 'not'
    text = re.sub(r'\'t', ' not', text)
    # add space after $
    text = text.replace(r'$', r'$ ')
    # remove explicit \n
    text = re.sub(r'(\\n)+', r' ', text)
    # replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)
    # remove @name references and hyperlinks
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', r'', text)
    # isolate and remove punctuations except '?'
    text = re.sub(r'([$\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
    text = re.sub(r'[^\w\s\?]', ' ', text)
    # remove multiple and trailing whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class AGNewsDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, header=None, names=['label', 'title', 'description'], index_col=False)
        self.data.dropna(inplace=True)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = (self.data['title'] + '. ' + self.data['description']).apply(preprocess)
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.data['label'] -= 1
        self.n_classes = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class AmazonDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, header=None, names=['label', 'title', 'description'], index_col=False)
        self.data.dropna(inplace=True)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = (self.data['title'] + '. ' + self.data['description']).apply(preprocess)
        self.data['label'] -= 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class DBPediaDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, header=None, names=['label', 'title', 'description'], index_col=False)
        self.data.dropna(inplace=True)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = (self.data['title'] + '. ' + self.data['description']).apply(preprocess)
        self.data['label'] -= 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 14

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class YahooDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, header=None, names=['label', 'q_title', 'q_content', 'best_answer'],
                                index_col=False)
        self.data.dropna(inplace=True)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = (self.data['q_title'] + self.data['q_content'] + self.data['best_answer']).apply(preprocess)
        self.data['label'] -= 1
        self.data.drop(columns=['q_title', 'q_content', 'best_answer'], inplace=True)
        self.n_classes = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class YelpDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, header=None, names=['label', 'text'], index_col=False)
        self.data.dropna(inplace=True)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['label'] -= 1
        self.data['text'] = self.data['text'].apply(preprocess)
        self.n_classes = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class AmazonShortDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, index_col=False)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = self.data['text'].apply(preprocess)
        self.n_classes = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class ImdbShortDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, index_col=False)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = self.data['text'].apply(preprocess)
        self.n_classes = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label


class YelpShortDataset(data.Dataset):
    def __init__(self, file_path, reduce):
        self.data = pd.read_csv(file_path, index_col=False)
        self.data = self.data.sample(reduce, random_state=42) if 0 < reduce < len(self.data) else self.data
        self.data['text'] = self.data['text'].apply(preprocess)
        self.n_classes = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return text, label
