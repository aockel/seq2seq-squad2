from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer



class Vocab:
    def __init__(self, name: str):
        self.name = name
        self.index = {"0": "<SOS>", "1": "<EOS>", "2": "<UNK>", "3": "<PAD>"}
        self.count = 4
        self.words = {"<SOS>": 0, "<EOS>": 1, "<UNK>": 2, "<PAD>": 3}
        self.wordcount = {"<SOS>": 9999999, "<EOS>": 9999999, "<UNK>": 9999999, "<PAD>": 9999999}

    # This function cleans our words before adding them
    @staticmethod
    def clean_text(text: str):
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"\s+", r" ", text).strip().lower()
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        return text

    def word2index(self, text):
        indexed_out = []
        for w in text:
            indexed_out.append(self.words.get(w, 2))
        return indexed_out

    def index2word(self, index):
        text_out = []
        for i in index:
            text_out.append(self.index.get(i, "<UNK>"))
        return text_out

    # This function indexes words in our vocabulary
    def index_word(self, word: str):
        if word not in self.words:
            self.words[word] = self.count
            self.index[str(self.count)] = word
            self.count += 1
        if word not in self.wordcount:
            self.wordcount[word] = 1
        else:
            self.wordcount[word] += 1

    def add_words(self, df):
        columns = list(df)
        count = 0
        for i in columns:
            for idx, r in df.iterrows():
                text = self.clean_text(r[i])
                for t in text:
                    if count % 15000 == 0:
                        print("Adding word {} to our vocabulary.".format(count))
                    self.index_word(t)
                    count += 1

    def get_top_word_counts(self):
        sorted_des = sorted(self.wordcount.items(), key=lambda x: x[1], reverse=True)
        return sorted_des[0:10] + sorted_des[-10:]

    def strip_low_word_counts(self, limit: int = 3):
        sorted_des = sorted(self.wordcount.items(), key=lambda x: x[1], reverse=True)
        to_drop = set([i[0] for i in sorted_des if i[1] < limit])
        for key in to_drop:
            idx = self.words[key]
            try:
                self.words.pop(key)
                self.index.pop(str(idx))
                self.wordcount.pop(key)
                self.count -= 1
            except:
                print(f'Err stripping word for key: {key} idx: {idx}')
        # re-index dict
        # generate new index
        self.index = dict(enumerate(self.index.values()))
        for key, value in self.index.items():
            self.words[value] = key









