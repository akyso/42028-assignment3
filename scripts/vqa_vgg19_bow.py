import pandas as pd
from keras.preprocessing.text import Tokenizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


training_df= pd.read_hdf('dataframe_train.h5', 'train')
valid_df = pd.read_hdf('dataframe_val.h5', 'val')

### Text Normalizing function. Part of the following function was taken from this link. 
def clean_text(text, user=None):
    ## Stemming
    text = text.lower().split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text

corpus = training_df['question'].append(valid_df['question'], ignore_index=True)
#corpus = corpus.append(pd.Series([np.nan]), ignore_index=True)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(corpus)

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

question_token_train = tokenizer.texts_to_sequences(training_df['question'])
question_token_val = tokenizer.texts_to_sequences(valid_df['question'])

def pad_tokens(df_serie, max_len=300):
    from keras.preprocessing.sequence import pad_sequences
    
    return pad_sequences(df_serie, padding='post', maxlen=max_len)

question_padded_token_train = pad_tokens(question_token_train, 26)
question_padded_token_val = pad_tokens(question_token_val, 26)

with open('question_padded_token_train.pkl', 'wb') as handle:
    pickle.dump(question_padded_token_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('question_padded_token_val.pkl', 'wb') as handle:
    pickle.dump(question_padded_token_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
