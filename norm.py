import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import RussianStemmer
#nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))
stop = []
for i in stop_words:
  if i not in ['нет','не','хорошо','иногда',
               'наконец','ни','никогда','ничего', 
               'опять','разве','совсем','уже']:
      stop.append(i)

morph = MorphAnalyzer()
stemmer = RussianStemmer(True)

def norm(line, stop_words=stop):
  line = line.lower()
  line = re.sub('[^ЙйёЁА-Яа-я0-9 ]', '', line)
  line = re.sub('\s+', ' ', line).strip()
  line = [i for i in line.split() if i not in stop_words]
  line = [stemmer.stem(morph.parse(word)[0].normal_form) for word in line]
  return ' '.join(line)


