#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[21]:


import os
os.chdir('C:\\Analytics\\MachineLearning\\Namo')


# In[22]:


data = pd.read_csv('modi-full-13-14.csv')


# In[23]:


data.head()


# In[24]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from nltk.sentiment.util import *

from nltk import tokenize

# sid = SentimentIntensityAnalyzer()


# In[25]:


tweets = pd.read_csv('modi-full-13-14.csv')


# In[26]:


tweets['sentiment_compound_polarity'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['compound'])
tweets['sentiment_neutral'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['neu'])
tweets['sentiment_negative'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['neg'])
tweets['sentiment_positive'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['pos'])
tweets['sentiment_type'] = ''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type'] = 'POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type'] = 'NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type'] = 'NEGATIVE'
tweets.sentiment_type.value_counts().plot(kind='bar',title='sentiment analysis')
plt.show()


# From the plot it is shown that 1000 positive tweets, 200 negative and positve tweets

# The above is the tweets before general election 2014

# In[27]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from nltk.sentiment.util import *

from nltk import tokenize


# In[28]:


tweets = pd.read_csv('modi-full-14-15.csv')


# In[29]:


tweets.head()


# In[30]:


tweets['sentiment_compound_polarity'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['compound'])
tweets['sentiment_neutral'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['neu'])
tweets['sentiment_negative'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['neg'])
tweets['sentiment_positive'] = tweets.Tweets.apply(lambda x:analyser.polarity_scores(x)['pos'])
tweets['sentiment_type'] = ''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type'] = 'POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type'] = 'NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type'] = 'NEGATIVE'
tweets.sentiment_type.value_counts().plot(kind='bar',title='sentiment analysis')
plt.show()


# Lets check the word cloud of PM Narendra Modi from 2013 to 2014 and then followed by 2014-2015

# In[31]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.subplot.bottom'] = 0.1

stopwords = set(STOPWORDS)
data = pd.read_csv('modi-full-13-14.csv')

wordcloud = WordCloud(
                    
                   background_color='white',
                   stopwords=stopwords,
                   max_words=100,
                   max_font_size=40,
                   random_state=42).generate(str(data['Tweets']))

print(wordcloud)
#fig.plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# From above world cloud we can see that he's trying to represent gujarat model to india, bjp and telling about advani etc before 2014 election

# In[34]:


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.subplot.bottom'] = 0.1

stopwords = set(STOPWORDS)
data = pd.read_csv('modi-full-14-15.csv')

wordcloud = WordCloud(
                    
                   background_color='white',
                   stopwords=stopwords,
                   max_words=100,
                   max_font_size=40,
                   random_state=42).generate(str(data['Tweets']))

print(wordcloud)
#fig.plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# After election he first visited to bhutan, Representing India to world, putting effort to make india better after 2014 election.

# Jai Hind

# In[ ]:




