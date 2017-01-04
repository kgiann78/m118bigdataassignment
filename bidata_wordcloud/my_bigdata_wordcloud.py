from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from os import listdir
import os.path
import string

if __name__ == "__main__":
    path = "../Dataset/"
    categories = []
    text = ""

    for category in listdir(path):
        categories.append(category)

    for category in categories:
        for f in listdir(path+category):
            temp_path = path+category+'/'+f
            if os.path.isfile(path+category+'/'+f):
                with open(temp_path, 'r') as f_curr:
                    s = f_curr.read()
                    text = text+" "+s

    word_cloud = WordCloud(stopwords=STOPWORDS).generate(text)

    plt.imshow(word_cloud)
    plt.axis("off")

    plt.savefig('worldcloud.png')
