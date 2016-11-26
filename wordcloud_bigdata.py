from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import glob
from os import listdir
import os.path
from operator import itemgetter

path = "Dataset/"
categories_counter = {}
text = ""

for category in listdir(path):
    categories_counter[category] = len(glob.glob(path+category+'/*'))

sorted_top5_categories = sorted(categories_counter.items(), key=itemgetter(1), reverse=True)
sorted_top5_categories = sorted_top5_categories[0:5]

for top5 in sorted_top5_categories:
    for f in listdir(path+top5[0]):
        temp_path = path+top5[0]+'/'+f
        if os.path.isfile(path+top5[0]+'/'+f):
            with open(temp_path, 'r') as f_curr:
                s = f_curr.read()
                text = text+" "+s


word_cloud = WordCloud(stopwords=STOPWORDS).generate(text)

plt.imshow(word_cloud)
plt.axis("off")

plt.show()
