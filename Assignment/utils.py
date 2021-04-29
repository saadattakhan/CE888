

import re

from emot.emo_unicode import UNICODE_EMO, EMOTICONS

import warnings
warnings.filterwarnings("ignore")

import string

class Utils:
    #### Natutal Language Processing With Spacy Library
    #### Named Entity Recognition and Removal
    #### Lemmatization
    #### Stop Words, Punctuation Removal
    #### Authentic Words
    @staticmethod
    def nlp_process(nlp,input_txt):
        doc = nlp(input_txt)
        for item in [(ent.text, ent.label_) for ent in doc.ents]:
                input_txt=input_txt.replace(item[0],'')
        #spacy_words = [t.lemma_ for t in doc if ((t.text in nlp.vocab) and (not t.is_stop) and (t.pos_ in ["ADJ","NOUN","PNOUN","VERB"]))]
        spacy_words = [t.lemma_ for t in doc if ((t.text in nlp.vocab) and (not t.is_stop) and (t.pos_ != "PUNCT"))]

        input_txt=" ".join(spacy_words)
        return input_txt

    #### Remove Extra Spaces
    @staticmethod
    def trim_text(text):
        return re.sub(' +',' ',text)

    #### Remove Retweets
    #### Remove URLs
    #### Remove Punctuation
    #### Remove Numbers and Digits
    #### Remove unnecessary words
    @staticmethod
    def clean_text(text):
        text=text.lower()
        text=re.sub("\S*@\S*\s?",'',text)
        text=re.sub(r"http\S+", "", text)
        ###punctuation
        text = re.sub(r'[^\w\s]',' ',text)
        ###numbers and digits
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
        text=re.sub('\w*\d\w*','', text)
        text=text.replace("[^a-zA-Z#]", " ")

        return text


    #### Convert Emojis to Textual Representation
    @staticmethod
    def convert_emojis(text):
        for emot in UNICODE_EMO:
            text = text.replace(emot, " ".join(UNICODE_EMO[emot].replace(",","").replace(":","").replace("_"," ").split()))
        return text

    #### Convert Emoticons to Textual Representation
    @staticmethod
    def convert_emoticons(text):
        for emot in EMOTICONS:
            text = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").replace("_"," ").split()), text)
        return text

    #### Get Hastags in Tweets
    @staticmethod
    def get_all_hashtags(input_txt):
        input_txt=input_txt.replace("#"," #")
        hashtags = [tag.strip("#") for tag in input_txt.split() if tag.startswith("#")]

        return hashtags

    #### Keep hastag without #
    @staticmethod
    def hashtag_handling(input_txt):
        hashtags=Utils.get_all_hashtags(input_txt)
        for tag in hashtags:
            input_txt=input_txt.replace("#"+tag,tag)
        return input_txt

    ##### Expand contractions
    @staticmethod
    def expand_contractions(text):
        contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not",
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

        # Regular expression for finding contractions
        contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    ##### Remove Twitter Handles
    @staticmethod
    def remove_handles(input_txt):
        r = re.findall("@[\w]*", input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)

        return input_txt

    ##### Remove short words
    @staticmethod
    def remove_short_words(input_txt):
        return ' '.join([w for w in input_txt.split() if len(w)>3])
