from Dataset import df
from Dataset import LabelEncoder,train_test_split
from Dataset import np
from Dataset import Tokenizer,sequence

x = df.v2
y = df.v1
le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.15)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words = max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequence_matrix = sequence.pad_sequences(sequences,maxlen = max_len)