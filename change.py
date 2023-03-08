import csv
#将tsv变为csv文件
with open('pacifier.tsv',encoding='utf-8') as f:
    data = f.read().replace('\t', ',')
with open('pacifier.csv','w',encoding='utf-8') as f:
    f.write(data)
f.close()
