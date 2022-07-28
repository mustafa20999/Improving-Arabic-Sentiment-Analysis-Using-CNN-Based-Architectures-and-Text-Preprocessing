import numpy as np 
import pandas as pd 

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import re
import warnings 
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "\path"]).decode("utf8"))



import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional,Conv2D
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
# Preprocessing
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
# Time
import time
import datetime
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.optimizers import Adam 
###########################
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
###########################
import string
import re
from nltk.corpus import stopwords
#########################
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical





# Date and time
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()

input_directory = r"../input/"
output_directory = r"../output/"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
    
figure_directory = "../output/figures"
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)
    
file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"



data = pd.read_csv('\path\dataset')
data.columns

data = data[['text', 'sentiment']]
data.head()

import nltk
nltk.download('punkt')

import nltk
nltk.download('stopwords')

import nltk
nltk.download('wordnet')

##The_stopword 
all_stopwords = stopwords.words('arabic')
sw_list = [
'إذ', 'إذا','إذما', 'اذن', 'إذن', 'أف', 'أقل', 'كتر', 'اجو', 'اتلم', 'كثر', 'اكثروا', 'اكثرة', 
'اكثرا', 'أكثر', 'الا', 'ألا', 'إلا', 'التي', 'الذي', 'الذين', 'اللاتي', 'اللائي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'إلى', 'إليك', 'إليكم', 'إليكما', 'إليكم', 'إليكن', 'أم', 'أما', 'أما', 'إما', 'آما', 'ان', 'أن',
 'إن', 'إنا', 'أنا', 'انا', 'انتي', 'أنتي', 'أنت', 'أنتت', 'أنتما', 'أنتم', 'أنتما', 'أنتن', 'أنما', 'إنما', 'إنه', 'أنني', 'آه', 'آها', 'أو', 'آوو', 'اوو ', 'او', 'أولاء', 'أولئك ', 'أوه', 'آي', 'أي', 'أيهم', 'أيتها', 'أيها', 'ايي', 
'إي', 'أين', 'أين', 'أينما', 'إيه', 'بخ', 'برا روح', 'بتاع', 'بب', 'بابا', 'بعدين', 'بر', 'بلاي', 'بح', 'بس', 'بعد', 'بعض', 'بك', 'بكي', 'بيكم', 'بكم', 'بكما', 'بكن', 'بل', 'بلى', 'بما', 'بماذا', 'بمن', 'بنا', 'به', 'بها',
 'بهم', 'بهما', 'بهن', 'بي', 'بين', 'بيد', 'تلك', 'تلكم', 'تلكما', 'ته', 'تي', 'تين', 'ثم', 'ثمة', 'جاب', 'جة', 'حط', 'خد', 'رابعه', 'راح', 'رح', 'شال', 'صح', 'طخ', 'عااا', 'كدا', 'ليل', 'حاشا', 'حبذا', 'حتى', 'حيث', 
' حيثما', 'حين', 'خلا', 'دون', 'ذا', 'ذات', 'ذاك', 'ذان', 'ذانك',
 'ذلك','ذلكم', 'ذلكما', 'ذلكن', 'ذه', 'ذو', 'ذوا', 'ذواتا', 'ذواتي', 'ذي', 'ذين', 'ذينك', 'ريثما', 'ريث', 'سوف', 'سوو', 'سواء', 'سوا', 'سوى ', 'شتان', 'عدا', 'عسى', 'عل', 'على', 'عبر', 'عنا', 'عليك', 'عليكي', 'عليكما',
 'عليكم', 'عليه', 'علا', ' عما', 'عن', 'عند', 'فإذا', 'فإن', 'فلا', 'فمن', 'في', 'فيم', ' فيما', 'فينكم', ' فينم', 'فين', 'فيهما', 'فيهم', 'فيه', 'فيها', ' قد', ' كأنهم', ' كأن', ' كأنما', ' كأي', ' كأين', ' وكدا', ' كذا', 'كذلك', ' كل',
 ' كلا', ' كلاهما', ' كلتا', ' كلما', ' كليكما', ' كليهما', ' كم', ' كما', ' كي', ' كيت', ' كيف', ' كيفما', ' لا', ' ليالي', ' لا لا لا', ' لسا', 'لاسيما', ' لدى', ' لسه', ' لستي', ' لست', ' لستما', ' لستم', ' لستما', 'لستن', ' لسن'
 ' لسنا', ' لعل', ' لك', ' لكم', ' لكما', ' لكن', ' لكنما', 'لكي', ' لكيلا', ' لم', ' لما', ' لن', ' لنا', ' له', ' لها', ' لهم', ' لهما', 'لهن', ' لو', ' لولا', ' لوما', ' لي', ' لئن', ' ليت', ' ليس', ' ليسا', ' ليست', ' ليستا',
 ' هع', ' هم', ' هاتان', ' هاته', ' هاتي', ' هاتين', ' هاك', ' هاهنا' , 'هذه', ' هذي', ' هذين', ' هكذا', ' هل', ' هلا', ' هم', ' هما', ' هن', ' هنا', 'هناك', ' هنالك', ' هو', ' هؤلاء', 'هي', ' هيا', ' هيت', 'هيهات', 
' والذين', ' وإذ', ' وإذا', ' وإن', ' ولا', ' ولكن', ' ولو', ' وما', 'ومن', ' وهو', ' وله', ' واد', ' وه', ' يا', ' ااا', ' أبريل', ' أبو', ' أبٌ ', ' أجل ', ' جمع ', ' أحد ', ' أخبر', ' أخذ', ' أخو ', ' أخٌ ', ' أربع ', 'أحمر', 
' أرح ', ' أربعا', ' أبكي', ' أرى', ' أسكن', ' أصبح', ' أصلا', ' أضحى', ' أطعم', ' أكتوبر', ' أعطى', ' أعلم', ' أغسطس', ' أفريل', ' أفعل به', ' أفٍّ ', ' ألف', ' ألفى', ' ألوف', ' أم', ' أما', 'أمام', ' أمامك', ' أمامكي', ' أمامكما', 
' أمامكم', ' أمامكَ', ' أمد', ' أمس', ' أمسى', ' مسا', ' امسا', ' تج', ' تحوّل', ' تحط', ' تخذ', ' ترك', ' تاني', ' تسع', ' تسعة', ' تسعين', ' تشرين', ' تعسا', ' تعلَّم', ' تفعلان', ' تفعلا', ' تفعلين', ' تفعلوا', ' تفعلون', ' تفعلين', 
' جانفي', ' جعل', ' جلل', ' جمعة', ' جميع', ' جنيه', ' جوان', ' جويلية',
 ' جير', ' ء', ' سابع', ' سادس', ' سبت', ' سبتمبر', ' سبحان', ' سبع', ' سبح', ' سير ', ' ست', ' ستة', ' ستكون', ' ستمئة', ' ستمائة', ' ستون', ' ستين', ' سحقا', ' سرا', ' سرعان', ' سقى ', ' سمعا', ' سنة', ' سنتيم', ' سنين', 
' سنوات', ' سوف', ' سوى', ' سين', ' صارِ', ' صباح', ' صبر', ' صبرا', ' صبرو', ' صدقا', ' صراحة', ' صفر', ' صهٍ', ' صهْ', ' ض', ' ضاد', ' ضحوة', ' ضد', ' ضامن', ' ضمن', ' ط', ' طاء', 'طاق', 'ّ طالما', ' طرا', ' طفق', 'ّ طَق', ' عشرة', 
' عشرات', 'عشرون', ' عشرين', ' عل', ' علق', ' علم', ' ه ', ' ما انفك', ' ما برح', ' مادام', ' مادامو'
, ' مادمنا', ' ماذا', ' ماما', ' متر', ' مرة',' مرات', ' مارس', ' مازال', ' مازالو', ' مازالنا', ' ماي', ' مايزال', ' مايزل', ' مايو', ' مافتئ', ' متىِ', ' مثل', ' مذ', ' مرات', ' مرّة', ' مساء', ' يفعلون', ' يكون', ' يلي', ' يمكن', ' يمين', ' ين', ' يناير', 
' يورو', ' يوليو', ' يومان', ' يونيو', ' يوان'
]
all_stopwords.extend(sw_list)





## pre-processing steps
'''
'''
    # Punctiations

punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation


stop_words = stopwords.words()

    #diacritics
arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

def preprocess(text ):
    
    '''
    Arabic text is entered as a string, and the preprocessed text is output. 
    '''
    
    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)
    

    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text )
    
    #words and corresponding emoticon
    text = re.sub("مرتبك", "o.o", text)
    text = re.sub("سعيد ", "(:", text)
    text = re.sub("وجهة سعيد جدا", "^_^", text)
    text = re.sub("غاضب", "):<", text)
    text = re.sub("بكاء", ")'：", text)
    text = re.sub("شيطاني", "(:3", text)
    text = re.sub("مللائكي", "o :) ", text)
    text = re.sub("سعيد", "8^) ", text)


    #remove normalization
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

  
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text




data['text'] = data['text'].apply(preprocess)
print(data.head(5))





#spilt and encode our data 
X = data.text	
Y = data.sentiment

label_encoder = LabelEncoder()

Y = label_encoder.fit_transform(Y)

Y = to_categorical(Y)



#split data & calculate maxword and max lenx 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 123)

max_words = len(set(" ".join(X_train).split()))
#max_len = X_train.apply(lambda x: len(x)).max()

#max_words = 2000
max_len = 100
#, 150, 120, 100
max_words, max_len


num_unique_word=max_words
MAX_REVIEW_LEN=max_len


#Hyperparameter value
max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
#batch_size = 500
epochs = 10
num_classes=2
embedding_size =128
# Convolution
kernel_size = 5
filters = 64
pool_size = 4


#Tokenize Text
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


#sequence padding 
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_test.shape)

maxlen=max_words




#Output Configuration
main_model_dir = output_directory + r"models/"
main_log_dir = output_directory + r"logs/"

try:
    os.mkdir(main_model_dir)
except:
    print("Could not create main model directory")
    
try:
    os.mkdir(main_log_dir)
except:
    print("Could not create main log directory")


model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')


try:
    os.mkdir(model_dir)
except:
    print("Could not create model directory")
    
try:
    os.mkdir(log_dir)
except:
    print("Could not create log directory")
    
model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"

print("Settting Callbacks")

checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    verbose=1)


callbacks = [checkpoint, reduce_lr, early_stopping]

callbacks = [early_stopping]

print("Set Callbacks at ", date_time(5))

def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


    

# word embedding
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):
    # word vectors
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    # embedding matrix
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(embeddings_index.values()) #for random init
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 
                                        (num_words, embed_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    max_features = embedding_matrix.shape[0]
    
    return embedding_matrix




#MC1 Model
model20= Sequential()
model20.add(Embedding(max_features, embedding_size, input_length=maxlen))
model20.add(Dropout(0.5))
model20.add(Conv1D(512, kernel_size=3,padding='same',activation='relu',strides=1))
model20.add(GlobalAveragePooling1D())
model20.add(Dropout(0.5))
model20.add(Dense(
    units=256,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)
))
model20.add(BatchNormalization())
model20.add(Dropout(0.5))
model20.add(Dense(2,activation='softmax'))
model20.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model20.summary()

loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
metrics = ['accuracy']

start_time = time.time()
print(date_time(1))
history3=model20.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=20, batch_size=batch_size, verbose=1)
elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print("\nElapsed Time: " + elapsed_time)
print("Completed Model Trainning", date_time(1))
#epochs=50,100,200,500

#Visualization
plot_performance(history=history3)



#MC2 Model
model4= Sequential()
model4.add(Embedding(max_features, embedding_size, input_length=maxlen))
model4.add(Conv1D(128,kernel_size=3,padding='same',activation='relu'))
model4.add(MaxPooling1D(pool_size=2))
model4.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
model4.add(MaxPooling1D(pool_size=2))
model4.add(Dropout(0.25))
model4.add(SpatialDropout1D(0.25))
model4.add(Bidirectional(GRU(128)))
model4.add(Dropout(0.5))
model4.add(Flatten())
model4.add(Dense(128,activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(2,activation='softmax'))
model4.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model4.summary()

loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
metrics = ['accuracy']

start_time = time.time()
print(date_time(1))
history3=model4.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=20, batch_size=batch_size, verbose=1)
elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print("\nElapsed Time: " + elapsed_time)
print("Completed Model Trainning", date_time(1))

#Visualization
plot_performance(history=history3)
