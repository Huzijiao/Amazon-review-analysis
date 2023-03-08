import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import re
import collections
import jieba	
import wordcloud	
from PIL import Image
def word_cloud(text, product_name):
	pattern1 = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"') 
	pattern2 = re.compile(r'<br />')
	pattern3 = re.compile('<BR>')
	text = re.sub(pattern1, '', text)
	text = re.sub(pattern2, ' ', text)
	text = re.sub(pattern3, ' ', text)
	# print(text)

	seg_list_exact = jieba.cut(text, cut_all = False)
	# print(seg_list_exact)
	object_list = []
	remove_words = [' ', 'the', 'I', ',', 'it', 'and', 'a', 'to', 'hair', "'"
	, 'is', 'my', 'this', 'for', '!', 'of', 'that', 'in', 'have', 'was', 'with'
	, 'It', 'but', 't', 'on', 'one', 'not', 's', 'so', 'as', 'The'
	, 'very', 'you', 'use', 'had', 'than', 'has', 'be', 'dry', 'just', 'out'
	, 'blow', 'at', 'time', 'product', 'This', 'used', 'me', 'when', 'can', 'cord'
	, 'about', 'are', 'really', 'or', 'only', 'becaues', 'works', 'get'
	, 'more', 'all', 'does','them','these','she','your','my','baby']
	for word in seg_list_exact:
	    if word not in remove_words:
	        object_list.append(word)

	word_counts = collections.Counter(object_list)
	word_counts_top10 = word_counts.most_common(100)
	classifier_f = open("naivebayes.pickle", "rb")
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	f = open(product_name + ".txt", "w")
	for item in word_counts_top10:
		class_ = classifier.prob_classify({item[0]:True})
		if  class_.prob(1) < 0.3:
			f.write(item[0] + ":" + str(item[1]) + "\n")
	f.close()
	#print (word_counts_top10)

	
	mask = np.array(Image.open('background.jpg')) 
	wc = wordcloud.WordCloud(
	     background_color='white',
	     mask=mask,
	     max_words=200, 
	     max_font_size=100 ,
	     scale=10  
	 )

	wc.generate_from_frequencies(word_counts) # 从字典生成词云
	image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
	wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
	wc.to_file("temp.jpeg") # 将图片输出为文件
	plt.imshow(wc) # 显示词云
	plt.axis('off') # 关闭坐标轴
	#plt.show() # 显示图像


def try_run(data, start, end):
	data = data[start:end]
	data.index=range(len(data))
	return data
product = pd.read_csv("pacifier.tsv", sep='\t')
product = try_run(product, 3457, 10000)
num = product.shape[0]
text = ""
for i in range(num):
    text += product.loc[i, "review_body"]
    #print(i)
    word_cloud(text,"pacifier")
