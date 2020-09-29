import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

spam_data = pd.read_csv(r"F:\projects\spam detection\spam.csv", engine='python')
spam_data = spam_data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
#check if there any nullset
sns.heatmap(spam_data.isnull(), yticklabels = False, cbar = False, cmap="Blues") #as we found out there's no cell with null data in dataframe

#visulize data
sns.countplot(x='v1',data = spam_data)
plt.show()
message_df = spam_data["v2"]
#visualizing the data
all_messages = message_df.tolist()
massive_string = ''.join(all_messages)
get_ipython().system('pip install WordCloud')
from wordcloud import WordCloud
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

from sklearn.feature_extraction.text import TfidfVectorizer    #for feature extraction
vectorizer_new = TfidfVectorizer(analyzer = cleaning_data)
spam_tfidfvectorizer_new  = vectorizer_new.fit_transform(spam_data["v2"])

X = spam_tfidfvectorizer_new
y = spam_data['v1']

#test_train_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


from sklearn.naive_bayes import GaussianNB

NB_classifier = GaussianNB()
NB_classifier_model = NB_classifier.fit(X_train.toarray(),y_train)

print("accuracy of test data prediction :",NB_classifier.score(X_test.toarray(),y_test))  #output is accuracy of the prediction
print(" ")
print("accuracy of train data prediction:",NB_classifier.score(X_train.toarray(),y_train))

from sklearn.metrics import classification_report, confusion_matrix
# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test.toarray())
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

##################   worst performance with this algorithm ##################


