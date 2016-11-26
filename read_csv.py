import os
import zipfile

path = 'Dataset/'

zf = zipfile.ZipFile('Datasets-2016.zip')
files = zf.open('train_set.csv')
files.readline()
for f in files.readlines():
    text = f.split('\t')

    if not os.path.exists(path+text[4]):
        os.makedirs(path+text[4])

    with open(path+text[4]+'/'+text[1], "w+") as data_set:
        data_set.write(text[2]+"\n"+text[3])
