import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

spam_data = pd.read_csv(r"spam.csv", engine='python')
print(spam_data)

spam_data = spam_data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

#check if there any nullset
sns.heatmap(spam_data.isnull(), yticklabels = False, cbar = False, cmap="Blues") #as we found out there's no cell with null data in dataframe

#visulize data
sns.countplot(x='v1',data = spam_data)
plt.show()

message_df = spam_data["v2"]
spam = spam_data[spam_data['v1']=='spam']

not_spam = spam_data[spam_data['v1']=='ham']

#visualizing the data
all_messages = message_df.tolist()
massive_string = ''.join(all_messages)

### pip install WordCloud
from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(massive_string))

#  not-spam messages
not_spam_mssges = not_spam["v2"].tolist()
not_spam_string = ''.join(not_spam_mssges)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(massive_string))

##### spam messages
spam_messages = spam["v2"].tolist()
spam_string = ''.join(spam_messages)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(massive_string))

## cleaning the data ###

#### for removing punctuation #####
import string
string.punctuation
### for removing stopwords #####

import nltk #natural langyage tool kit
from nltk.corpus import stopwords


## stopwords
#nltk.download('punkt')
#nltk.download('wordnet')
nltk.download('stopwords')
stopwords.words('english')

# for lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

### for stemming
from nltk.stem import SnowballStemmer   # for stemming
snowball_stemmer = SnowballStemmer('english')  #for stemming


### for removing numbers #####
def cleaning_data(messages):
    #removing punctuation
    msg_with_removed_punc = ''.join([char for char in messages if char not in string.punctuation])
    #removing numbers
    msg_with_removed_num = ''.join([char for char in msg_with_removed_punc if char not in '1234567890'])
    #convert from uppercase to lowercase
    msg_aftr_converted_to_Lowercase = ''.join([char.lower() for char in msg_with_removed_num])
    #lemmatization
    lem_word_tokens = nltk.word_tokenize(msg_aftr_converted_to_Lowercase)
    lemmatized_message = ''.join([wordnet_lemmatizer.lemmatize(word) for word in lem_word_tokens ])
    #stemming
    stemming_word_tokens = nltk.word_tokenize(lemmatized_message)
    stemmed_message = ''.join([snowball_stemmer.stem(word) for word in stemming_word_tokens])
    #stop words
    stopwords_tokens = nltk.word_tokenize(stemmed_message)
    msg_with_removed_stopwords = ''.join([word for word in stopwords_tokens if word not in stopwords.words('english')])
    return msg_with_removed_stopwords

new_data = spam_data['v2'].apply(cleaning_data)

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#feature extraction with countvectorizer
vectorizer = CountVectorizer(analyzer = cleaning_data)
spam_countvectorizer = vectorizer.fit_transform(spam_data['v2'])
#print(vectorizer.get_feature_names())


################ feature extraction with tfidfvectorizer  #############
#tf_idfvectorizer = TfidfVectorizer(analyzer = cleaning_data)
#spam_tfidfvectorizer = tf_idfvectorizer.fit_transform(spam_data['v2'])
#print(tweets_tfidfvectorizer.get_feature_names())
#X = spam_tfidfvectorizer


X = spam_countvectorizer
y = spam_data['v1']

#test_train_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier_model = NB_classifier.fit(X_train,y_train)


print("accuracy of test data prediction2 :",NB_classifier_model.score(X_train,y_train,sample_weight=None),'\n') #0.940
print("accuracy of test data prediction2 :",NB_classifier_model.score(X_test,y_test,sample_weight=None))        #0.946

from sklearn.metrics import classification_report, confusion_matrix

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))

### better accuracy achieved with counvectorizer + multinomialNB algorithm