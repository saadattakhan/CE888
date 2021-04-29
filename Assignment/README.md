## Twitter Analysis
### Hate Speech Detection | Offensive Speech Detection | Sentiment Analysis

#### Download dataset
[LINK](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/hate) to Hate Speech Dataset <br />
[LINK](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/offensive) to Offensive Speech dataset<br />
[LINK](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment) to Sentiment Analysis dataset<br />

#### Install Virtual Environment
pip install --user virtualenv
#### Create Virtual Environment
python -m venv tenv
#### Activate Virtual Environment
source env/bin/activate
#### Install Requirements
pip install -r requirements.txt 

#### Run Code with dataset path and task name (hate,offensive,sentiment)
python main.py 'hate/' 'hate'
