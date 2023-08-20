from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchtext
from vocab import Vocab


def load_df(max_rows: int = 0):
    """
    Loads the dataset into a Pandas Dataframe for processing.
    :parameter max_rows: train data set contains 86821 rows. The dev set 5928 which can not be limited.
    :returns df_train, df_dev
    """
    # create Dictionary objects of question and answer
    dict_train = { "question": [], "answer": []}
    dict_dev = {"question": [], "answer": []}
    # fetch dataset
    train, dev = torchtext.datasets.SQuAD2(root='./data', split=('train', 'dev'))
    # iterate over training set
    cnt_rows = 0
    for c, question, answer, i in train:
        if answer[0]:
            dict_train["question"].append(question)
            dict_train["answer"].append(answer[0])
            cnt_rows += 1
        if 0 < max_rows <= cnt_rows:
            break
    df_train = pd.DataFrame.from_dict(dict_train)
    # iterate over dev set
    for c, question, answer, i  in dev:
        if answer[0]:
            dict_dev["question"].append(question)
            dict_dev["answer"].append(answer[0])
    df_dev = pd.DataFrame.from_dict(dict_dev)
    print(f'Train data frame contains {len(df_train.index)} rows.')
    print(f'Dev data frame contains {len(df_dev.index)} rows.')
    return df_train, df_dev


def prepare_text(max_rows_train_set: int = 0, count_limit: int = 5, min_length: int = 3, max_length: int = 13, stage: str = 'dev'):
    """
    Creates a vocab object and cleans and tokenizes the question and answer.
    :var
        max_rows_train_set: limit the number of rows in the train data set. Train data set contains 86821 rows.
        count_limit: remove words that occur less than count_limit times in the text, default=5
        min_length: Minimum length of question or answer, default = 3
        max_length: Maximum length of question or answer, default=13
        stage: Prepare vocab and tokenized df for one of dev|train, default=dev
    :returns
        vocab: vocabulary object
        tokenized: tokenized dataframe
    """

    my_vocab = Vocab(name='qna')
    # load data
    train_df, dev_df = load_df(max_rows=max_rows_train_set)
    # add cleaned words to vocab
    if stage == 'dev':
        text_df = dev_df
    else:
        text_df = train_df
    # add vocab
    my_vocab.add_words(text_df)
    # strip words that have only rare occurrence
    cnt_in = my_vocab.count
    my_vocab.strip_low_word_counts(limit=count_limit)
    cnt_out = my_vocab.count
    print(f'Word count in vocab is now {cnt_out}, removed {cnt_in-cnt_out} words during cleanup.')
    # clean df and remove text that does contain words that have been stripped from vocab
    print(f'Data frame contains {len(text_df.index)} rows.')
    text_df['q_tokens'] = text_df.question.apply(my_vocab.clean_text)
    text_df['a_tokens'] = text_df.answer.apply(my_vocab.clean_text)
    for idx, r in text_df.iterrows():
        # clean the sentences
        q_clean = r['q_tokens']
        a_clean = r['a_tokens']
        remove_qa = False
        for word in q_clean:
            if word not in my_vocab.words:
                remove_qa = True
                break
        for word in a_clean:
            if word not in my_vocab.words:
                remove_qa = True
                break
        # remove long questions or answers (adding sos and eos will add 2 tokens)
        if len(q_clean) >= max_length-2 or len(a_clean) >= max_length-2:
            remove_qa = True
        # remove short questions or answers
        if len(q_clean) < min_length or len(a_clean) < min_length:
            remove_qa = True
        if remove_qa:
            text_df.drop(index=idx, inplace = True)
    print(f'Data frame after row cleanup contains {len(text_df.index)} rows.')
    # create indexed sentences
    def prep_index(sentence: str):
        # start with SOS token (==0 in vocab)
        indexed_sentence = [0]
        indexed_sentence += [my_vocab.words.get(w, 2) for w in sentence]
        # end with EOS token (==1 in vocab)
        indexed_sentence += [1]
        # pad with <PAD> token (==3 in vocab)
        idx_len = len(indexed_sentence)
        if idx_len < max_length:
            indexed_sentence += [3] * (max_length-idx_len)
        return indexed_sentence
    # tokenize the questions and answer column
    text_df['SRC'] = text_df.q_tokens.apply(prep_index)
    text_df['TRG'] = text_df.a_tokens.apply(prep_index)
    text_df['SRC'].to_numpy()
    text_df['TRG'].to_numpy()
    text_df.reset_index(drop=True, inplace=True)
    return my_vocab, text_df


def train_test_split(input_df, fraction_train: float = 0.7, fraction_test_val: float = 0.75):
    """
    Input:
        input_df
        fraction: the fraction of data that should be used for the training set

    Output:
        train_out:
        test_out:
        valid_out
    """
    train_set = input_df.sample(frac=fraction_train, random_state=200)
    rest_set = input_df.drop(train_set.index)
    rest_set.reset_index(drop=True, inplace=True)
    test_set = rest_set.sample(frac=fraction_test_val, random_state=100)
    valid_set = rest_set.drop(test_set.index)

    train_out = train_set[['SRC', 'TRG']]
    test_out = test_set[['SRC', 'TRG']]
    valid_out = valid_set[['SRC', 'TRG']]
    train_out.reset_index(drop=True, inplace=True)
    test_out.reset_index(drop=True, inplace=True)
    valid_out.reset_index(drop=True, inplace=True)
    print(f'Train set of length: {len(train_out)}')
    print(f'Test set of length: {len(test_out)}')
    print(f'Valid set of length: {len(valid_out)}')
    return train_out, test_out, valid_out


def get_dataloader(train_set, test_set, valid_set, batch_size: int):
    """
    create data loader iterators

    :param train_set: training data
    :param test_set: test data
    :param valid_set: validation data
    :param batch_size: batch size for DataLoader
    :return: train_dataloader
    :return: test_dataloader
    :return: valid_dataloader
    """
    class QnADataset(Dataset):
        def __init__(self, SRC, TRG):
            self.source = SRC
            self.target = TRG

        def __len__(self):
            return len(self.source)

        def __getitem__(self, idx):
            # sample = {"SRC": self.source[idx], "TRG": self.target[idx]}
            return self.source[idx], self.target[idx]

    # create my dataset from pandas df
    train_d = QnADataset(train_set['SRC'], train_set['TRG'])
    # print('\nFirst iteration of data set: ', next(iter(TrainD)), '\n')
    test_d = QnADataset(test_set['SRC'], test_set['TRG'])
    valid_d = QnADataset(valid_set['SRC'], valid_set['TRG'])

    def collate_batch(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return torch.Tensor(data).to(torch.int64), torch.Tensor(target).to(torch.int64)

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_d, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch)
    print(f'Created `train_dataloader` with {len(train_dataloader)} batches!')
    test_dataloader = DataLoader(test_d, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch)
    print(f'Created `test_dataloader` with {len(test_dataloader)} batches!')
    valid_dataloader = DataLoader(valid_d, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch)
    print(f'Created `test_dataloader` with {len(valid_dataloader)} batches!')

    return train_dataloader, test_dataloader, valid_dataloader
