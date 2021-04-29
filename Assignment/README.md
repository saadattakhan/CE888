## Twitter Analysis
### Hate Speech Detection | Offensive Speech Detection | Sentiment Analysis
In this project I evaluated multiple Machine Learning Models on Twitter dataset for classification task. I have performed necessary preprocessing for cleaning tweets data and evaluated Logistic Regression Classifier, Linear SVC Classifier, Random Forest Classifier and XGB Classifier on test dataset. I achieved following F-1 score for three classification tasks

Task | Classifier | F-1 Score (Test) | 
--- | --- | --- | 
Hate Speech Detection | Random Forest | 0.57 | 
Offensive Speech Identification | Random Forest | 0.71 | 
Sentiment Analysis | Logistic Regression | 0.55 | 

[LINK](https://github.com/saadattakhan/CE888/blob/main/Assignment/Twitter%20Analysis%20for%20Hate%2C%20Offensive%20and%20Sentiment.pdf) to paper

#### Download dataset
[LINK](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/hate) to Hate Speech Dataset <br />
[LINK](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/offensive) to Offensive Speech dataset<br />
[LINK](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment) to Sentiment Analysis dataset<br />

#### Install Virtual Environment
pip install --user virtualenv
#### Create Virtual Environment
python -m venv tenv
#### Activate Virtual Environment
source tenv/bin/activate
#### Install Requirements
pip install -r requirements.txt

#### Install Spacy Dataset
python -m spacy download en_core_web_sm<br />
python -m spacy download en_core_web_md

#### Run Code with dataset path and task name (hate,offensive,sentiment)
python main.py 'hate/' 'hate'
