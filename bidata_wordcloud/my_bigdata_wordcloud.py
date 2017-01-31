from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import zipfile
import pandas as pd

if __name__ == "__main__":

    #
    #   Prepare data
    #

    cloud = ""

    zf = zipfile.ZipFile('../Datasets-2016.zip')
    files = zf.open('train_set.csv')
    df = pd.read_csv(files, sep='\t',
                     names=['RowNum', 'Id', 'Title', 'Content', 'Category'])

    # Concatenate Title and Content into text
    df['text']=df['Title'].map(str)+df['Content']

    # Create a single document of words
    for i in range(1, df['text'].count()):
        cloud = cloud+" "+df['text'][i]

    #
    #   Apply WordCloud to the document
    #
    word_cloud = WordCloud(stopwords=STOPWORDS).generate(cloud)

    #
    #   Create image of WordCloud
    #

    plt.imshow(word_cloud)
    plt.axis("off")

    plt.savefig('worldcloud.png')
