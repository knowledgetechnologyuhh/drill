import os
from shutil import rmtree
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
import pandas as pd
from sklearn import model_selection

from datasets.cls_dataset import AGNewsDataset, AmazonDataset, DBPediaDataset, YahooDataset, YelpDataset, \
    AmazonShortDataset, ImdbShortDataset, YelpShortDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but adds get_idxs functionality
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx


def load_dataset(base_path, name, reduce_train, reduce_test):
    if name.upper() == 'AGNEWS':
        train = AGNewsDataset(os.path.join(base_path, 'data/ag_news_csv/train.csv'), reduce_train)
        test = AGNewsDataset(os.path.join(base_path, 'data/ag_news_csv/test.csv'), reduce_test)
    elif name.upper() == 'AMAZON':
        train = AmazonDataset(os.path.join(base_path, 'data/amazon_review_full_csv/train.csv'), reduce_train)
        test = AmazonDataset(os.path.join(base_path, 'data/amazon_review_full_csv/test.csv'), reduce_test)
    elif name.upper() == 'YELP':
        train = YelpDataset(os.path.join(base_path, 'data/yelp_review_full_csv/train.csv'), reduce_train)
        test = YelpDataset(os.path.join(base_path, 'data/yelp_review_full_csv/test.csv'), reduce_test)
    elif name.upper() == 'DBPEDIA':
        train = DBPediaDataset(os.path.join(base_path, 'data/dbpedia_csv/train.csv'), reduce_train)
        test = DBPediaDataset(os.path.join(base_path, 'data/dbpedia_csv/test.csv'), reduce_test)
    elif name.upper() == 'YAHOO':
        train = YahooDataset(os.path.join(base_path, 'data/yahoo_answers_csv/train.csv'), reduce_train)
        test = YahooDataset(os.path.join(base_path, 'data/yahoo_answers_csv/test.csv'), reduce_test)
    elif name.upper() == 'AMAZONSHORT' or name.upper() == 'AMAZON_S':
        dir_path = os.path.join(base_path, 'data/uci_sentiment')
        if not os.path.exists(os.path.join(dir_path, 'amazon')):
            download_uci_sentiment(dir_path)
        train = AmazonShortDataset(os.path.join(dir_path, 'amazon/train.csv'), reduce_train)
        test = AmazonShortDataset(os.path.join(dir_path, 'amazon/test.csv'), reduce_test)
    elif name.upper() == 'IMDBSHORT' or name.upper() == 'IMDB_S':
        dir_path = os.path.join(base_path, 'data/uci_sentiment')
        if not os.path.exists(os.path.join(dir_path, 'imdb')):
            download_uci_sentiment(dir_path)
        train = ImdbShortDataset(os.path.join(dir_path, 'imdb/train.csv'), reduce_train)
        test = ImdbShortDataset(os.path.join(dir_path, 'imdb/test.csv'), reduce_test)
    elif name.upper() == 'YELPSHORT' or name.upper() == 'YELP_S':
        dir_path = os.path.join(base_path, 'data/uci_sentiment')
        if not os.path.exists(os.path.join(dir_path, 'yelp')):
            download_uci_sentiment(dir_path)
        train = YelpShortDataset(os.path.join(dir_path, 'yelp/train.csv'), reduce_train)
        test = YelpShortDataset(os.path.join(dir_path, 'yelp/test.csv'), reduce_test)
    else:
        raise Exception('The requested dataset name {} is not valid. Check typos or implementation.'.format(name))

    return train, test


def offset_labels(dataset):
    # Amazon / Yelp
    offset_by = 0
    # AGNews
    if isinstance(dataset, AGNewsDataset):
        offset_by = 5
    # DBPedia
    elif isinstance(dataset, DBPediaDataset):
        offset_by = 5 + 4
    # YahooAnswers
    elif isinstance(dataset, YahooDataset):
        offset_by = 5 + 4 + 14
    # UCI Sentiment Labelled Sentences Datasets (can be used for sanity checks)
    elif isinstance(dataset, YelpShortDataset):
        offset_by = 0
    elif isinstance(dataset, AmazonShortDataset):
        offset_by = 2
    elif isinstance(dataset, ImdbShortDataset):
        offset_by = 4

    dataset.data['label'] = dataset.data['label'] + offset_by
    return dataset


def download_uci_sentiment(dir_path, test_size=0.3):
    """
    Loads UCI Sentiment Labelled Sentences database, splits into train/test and saves it as .csv files
    :param dir_path: Path to target directory where train and test files will be stored.
    :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include
    in the test split. If int, represents the absolute number of test samples.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip'
    with urlopen(url) as r:
        with ZipFile(BytesIO(r.read())) as zfile:
            files = [zfile.extract(file, dir_path) for file in zfile.namelist()
                     if file.startswith('sentiment') and file.endswith('labelled.txt')]

    for (name, file) in zip(['amazon', 'imdb', 'yelp'], files):
        os.makedirs(os.path.join(dir_path, name), exist_ok=True)
        df = pd.read_csv(file, header=None, sep='\t', names=['text', 'label'], index_col=False) if name != 'imdb' \
            else pd.read_csv(file, header=None, sep=' \t', names=['text', 'label'], index_col=False)
        train, test = model_selection.train_test_split(df, random_state=42, test_size=test_size, stratify=df['label'])
        train.to_csv(os.path.join(dir_path, name, 'train.csv'), header=True, index=False)
        test.to_csv(os.path.join(dir_path, name, 'test.csv'), header=True, index=False)
    rmtree(os.path.join(dir_path, 'sentiment labelled sentences'))


def batch_encode(batch):
    text, labels = [], []
    for txt, lbl in batch:
        text.append(txt)
        labels.append(lbl)
    return text, labels
