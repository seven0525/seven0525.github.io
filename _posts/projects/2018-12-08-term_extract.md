---
layout: post
title:  "TermExtractを使ってツートからキーワードを抽出"
date:   2018-12-08
excerpt: "TermExtractとTwitterAPIで、指定したユーザーに関連したキーワードを抽出します"
project: true
tag:
- Machine Larning
- Twitter
- TermExtract
- API
comments: false
---

# TermExtractを使ってツートからキーワードを抽出

## ユーザーを指定して取得した400件のツイートのうち名詞だけを取り出す。

```py
#!/usr/bin/env python                                                                                                                                             
# -*- coding:utf-8 -*-  
import json
from requests_oauthlib import OAuth1Session
from twitter import Twitter, OAuth
from janome.tokenizer import Tokenizer
import collections
import re
from collections import Counter, defaultdict
import sys, json, time, calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta

#APIキーの設置
CONSUMER_KEY =  'YOURS'
CONSUMER_SECRET = 'YOURS'
ACCESS_TOKEN = 'YOURS'
ACCESS_SECRET = 'YOURS'

t = Twitter(auth=OAuth(
    ACCESS_TOKEN,
    ACCESS_SECRET,
    CONSUMER_KEY,
    CONSUMER_SECRET
))
    
twitter = OAuth1Session(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
url = "https://api.twitter.com/1.1/search/tweets.json"
userTweets = []


def get_userstweets_again(screen_name, max_id):
    
    max_id = max_id
    count = 200 #一度のアクセスで何件取ってくるか
    aTimeLine = t.statuses.user_timeline(screen_name = screen_name, count=count, max_id=max_id)
    for tweet in aTimeLine:
        userTweets.append(tweet['text'])
            


def get_userstweets(screen_name):
    number_of_tweets = 0
    count = 200 #一度のアクセスで何件取ってくるか
    aTimeLine = t.statuses.user_timeline(screen_name = screen_name, count=count, include_rts='false',)
    for tweet in aTimeLine:
        number_of_tweets += 1
        userTweets.append(tweet['text'])
        if number_of_tweets >= 200:
            max_id = tweet["id"]
            print(max_id)
            get_userstweets_again(screen_name, max_id)

#検索したい相手を指定 
print("相性を調べたい相手のuser_idを入力してください")
target_user_id = input('>> ')
print('----------------------------------------------------')

get_userstweets(target_user_id)

print(userTweets)
print(len(userTweets))
```


## TermExtractで取得したうちから複合語を作る

```py
# #複合語をつくる
from janome.tokenizer import Tokenizer
import collections
import re
import sys
import MeCab
import collections
import termextract.japanese_plaintext
import termextract.core

for text in userTweets:
    if "RT" in text:
        userTweets.remove(text)
        
texts = ','.join(userTweets)

texts=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", texts)
texts=re.sub('RT', "", texts)
texts=re.sub('お気に入り', "", texts)
texts=re.sub('まとめ', "", texts)
texts=re.sub(r'[!-~]', "", texts)#半角記号,数字,英字
texts=re.sub(r'[︰-＠]', "", texts)#全角記号
texts=re.sub('\n', "", texts)#改行文字

# 複合語を抽出し、重要度を算出
frequency = termextract.japanese_plaintext.cmp_noun_dict(texts)
LR = termextract.core.score_lr(frequency,
         ignore_words=termextract.japanese_plaintext.IGNORE_WORDS,
         lr_mode=1, average_rate=1
     )
term_imp = termextract.core.term_importance(frequency, LR)

# 重要度が高い順に並べ替えて出力
data_collection = collections.Counter(term_imp)
for cmp_noun, value in data_collection.most_common():
    print(termextract.core.modify_agglutinative_lang(cmp_noun), value, sep="\t")
```

自分（@ahpjop）の場合、上からこのような結果になりました。

時間	51.81650944742986  
今日	39.96088014880467  
明日	29.30273201187765  
自分	18.402613070329647  
参加	15.004664490468443  
参加者	13.862414010542652  
完全	13.401475835338031  
松屋好	10.78607775151476  
平日	10.360080256445405  
ハッカソン	8.485281374238571  
理解	8.34947305114122  
読書	7.977443845417483  
日食	7.745966692414834  
今回時間	7.4582845037286365  
募集	7.325683002969413  
参加者人	6.774295719813265  
時間寝	6.6240812509706535  
当日	6.619501839293746  
後半時間	6.462224637707511  
無限	6.4474195909412515  
日本語	6.214465011907717  
時間帯	6.191232418226783  
時間プログラミング	6.191232418226783  