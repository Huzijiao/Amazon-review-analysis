#Clean Data
import numpy as np
import pandas as pd
hair_dryer = pd.read_csv('hair_dryer.tsv', sep ='\t')
microwave = pd.read_csv('microwave.tsv', sep ='\t')
pacifier = pd. read_csv('pacifier.tsv', sep ='\t')
def washer(dataset):
    new_dataset = dataset.drop (['marketplace','product_category'], axis =1)
    new_dataset .loc[ new_dataset ['vine'] =='n','vine'] ='N'
    new_dataset .loc[ new_dataset ['vine'] =='y','vine'] ='Y'
    new_dataset .loc[ new_dataset ['verified_purchase'] =='n','verified_purchase'] ='N'
    new_dataset .loc[ new_dataset ['verified_purchase'] =='y','verified_purchase'] ='Y'

    transfer = [('','\''), ('','\''), ('','"'), ('','"'), ('',','), ('','.'), ('','!'), ('','...'), ('','-'), ('','-')]
    accept ='[^a-zA -Z0 -9_ !? ,.\'" ‘+ -=;() \[\] < >*#~&$^@%/|\\\\]'
    for pair in transfer:
        new_dataset ['product_title'] = new_dataset ['product_title'].str.replace(pair[0], pair [1])
        new_dataset ['review_headline'] = new_dataset ['review_headline'].str.replace(pair [0], pair [1])
        new_dataset ['review_body'] = new_dataset ['review_body'].str.replace(pair [0],pair [1])
    trash = new_dataset [ new_dataset ['product_title'].str.contains(accept , na=True)+ new_dataset ['review_headline'].str.contains(accept , na=True)+ new_dataset ['review_body'].str.contains(accept , na=True)]
    new_dataset = new_dataset [- ( new_dataset ['product_title'].str.contains(accept , na=True)+ new_dataset ['review_headline'].str.contains(accept , na=True)+ new_dataset ['review_body'].str.contains(accept , na=True))]
    return new_dataset , trash

new_hair_dryer , trash1 = washer(hair_dryer )
new_microwave , trash2 = washer(microwave )
new_pacifier , trash3 = washer(pacifier)
trash = pd.concat ([ trash1 , trash2 , trash3])

new_hair_dryer .to_csv('data/ new_hair_dryer .csv', encoding ='utf -8 _sig')
new_microwave .to_csv('data/ new_microwave .csv', encoding ='utf -8 _sig')
new_pacifier .to_csv('data/ new_pacifier .csv', encoding ='utf -8 _sig')
trash.to_csv('data/trash.csv', encoding ='utf -8 _sig')
del hair_dryer , microwave , pacifier , trash1 , trash2 , trash3
### 1. Prepare: Import , Time Process , Review Process
import matplotlib .pyplot as plt
def data_process (dataset):
    dataset['review_date'] = pd. to_datetime (dataset['review_date'])
    dataset['year'] = dataset['review_date'].dt.year
    dataset['month'] = dataset['review_date'].dt.month
    dataset['day'] = dataset['review_date'].dt.day
    dataset['helpful_rate'] = dataset['helpful_votes'] / dataset['total_votes']
    dataset['review_length'] = dataset['review_headline'].str.len() + dataset[' review_body'].str.len()
data_process ( new_hair_dryer )
data_process ( new_microwave )
data_process ( new_pacifier )
all = pd.concat ([ new_hair_dryer , new_microwave , new_pacifier ])
# ######################################################
### 2.1. Plot 1
import calendar
legend = ['all','hair_dryer','microwave','pacifier']
review_num_to_date = pd.concat ([all.groupby (('year','month'))['star_rating']. count (),
new_hair_dryer .groupby (('year','month'))['star_rating']. count (),
new_microwave .groupby (('year','month'))['star_rating']. count (),
new_pacifier .groupby (('year','month'))['star_rating'].count ()], axis =1)
review_num_to_date .columns = legend
review_num_to_date = review_num_to_date . reset_index ()
review_num_to_date ['date'] = review_num_to_date ['year'] + review_num_to_date ['month'] / 12
review_num_to_date = review_num_to_date .drop (['year','month'], axis =1).set_index ('date')
review_num_to_date .plot(legend=True , alpha =0.5)
# plt.title('Review Amount per Month')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.savefig("figure/Review Amount per Month.png",dpi =500 , bbox_inches ='tight')
plt.show ()

def func(dataset):
    series = dataset.groupby (('year','month'))['star_rating']. agg (['count','mean'])
    series[series['count']<3] = None
    series = series. reset_index ()
    series['date'] = series['year'] + series['month'] / 12
    series = series.drop (['year','month','count'], axis =1).set_index ('date')
    return series
ser_all , ser_hair , ser_micro , ser_paci = func(all), func( new_hair_dryer ), func(
new_microwave ), func( new_pacifier )
review_num_to_date = pd.concat ([ ser_hair , ser_micro , ser_paci], axis =1)
review_num_to_date .columns = legend [1:]
review_num_to_date .plot(legend=True , alpha =0.5)
# plt.title('Rating Score per Month')
plt.xlabel('Date')
plt.ylabel('Score')
# plt.xticks(np.arange (0, 168, 30))
plt.savefig("figure/Rating Score per Month.png",dpi =500 , bbox_inches ='tight')
plt.show ()

del legend , review_num_to_date , ser_all , ser_hair , ser_micro , ser_paci
# ######################################################
### 2.2. Plot 2
 # ######################################################


review_percent_to_rating = pd.concat ([all.groupby('star_rating')['star_rating']. count ()/ all.shape [0],new_hair_dryer .groupby('star_rating')['star_rating'].count () / new_hair_dryer .shape [0],new_microwave .groupby('star_rating')['star_rating'].count () / new_microwave .shape [0],new_pacifier .groupby('star_rating')['star_rating'].count () / new_pacifier .shape [0]] , axis =1)
review_percent_to_rating .columns = ['all','hair_dryer','microwave','pacifier']
review_percent_to_rating .plot.barh(legend=True , alpha = 0.5)
# plt.title('Review Percent per Rating')
plt.xlabel('Percent')
plt.ylabel('Rating')
plt.savefig("figure/Review Percent per Rating.png",dpi =500 , bbox_inches ='tight')
plt.show ()

del review_percent_to_rating
### 2.3. Plot 3
for can in [all, new_hair_dryer , new_pacifier , new_microwave ]:
    can[can['total_votes'] > 10]['helpful_rate']. plot.hist(bins =25)
# plt.title('Review Amount of Helpful Percent')
plt.xlabel('Helpful Percent')
plt.ylabel('Amount')
plt.legend (['all','hair_dryer','pacifier','microwave'])
plt.savefig("figure/Review Amount of Helpful Percent.png",dpi =500 , bbox_inches ='tight')
plt.show ()

del can

# ######################################################
### 2.4. Plot 4
# ######################################################

all.groupby('customer_id')['customer_id']. agg ({'num':'count'}). reset_index ().groupby(' num').count ().plot(kind='bar', logy=True , legend=False , alpha =0.7)
# plt.title('Customer Amount for Review Number')
plt.xlabel('Review Number')
plt.ylabel('Customer Amount')
plt.savefig("figure/Customer Amount for Review Number.png",dpi =500 , bbox_inches ='tight')
plt.show ()


# ######################################################
### 2.5. Plot 5
# ######################################################
def trans(k):
    if k >=0:
        return int(k*50) /50
    return None

review_length = all[all['total_votes'] >10][['helpful_rate','review_length']]
review_length ['helpful_level'] = pd.Series ([ trans(k) for k in review_length [' helpful_rate']], index= review_length .index)
review_length .groupby('helpful_level')['review_length']. agg ({'average_length':'mean'}).plot ()
plt.scatter( review_length ['helpful_rate'], review_length ['review_length'], s=(2 ,), c='#ff00ff', alpha =0.3)
# plt.title('Average Review Length of Helpful Percent')
plt.xlabel('Helpful Percent')
plt.ylabel('Review Length')
plt.legend (['Average Review Length'])
plt.savefig("figure/Average Review Length of Helpful Percent.png",dpi =500 , bbox_inches =
'tight')
plt.show ()
star_rating = all[all['total_votes'] >10][['helpful_rate','star_rating']]. groupby(' star_rating')['helpful_rate']. mean ()

del review_length

# ######################################################
### 2.6. Plot 6
# ######################################################

star_rating = all[all['total_votes'] >10][['helpful_rate','star_rating']]
star_rating ['helpful_level'] = pd.Series ([ trans(k) for k in star_rating ['helpful_rate']], index= star_rating .index)
star_rating .groupby('helpful_level')['star_rating']. agg ({'Average Star Rating':'mean'}).plot(alpha =0.7)
helpful_star_frequency = star_rating .groupby (['helpful_level','star_rating']).count ().reset_index ()
helpful_star_frequency .columns = ['helpful_level','star_rating','freqency']
plt.scatter( helpful_star_frequency ['helpful_level'], helpful_star_frequency [' star_rating'], helpful_star_frequency ['freqency']*10 ,c='#ff00ff', alpha =0.3)

# plt.title('Star Rating of Helpful Percent')
plt.xlabel('Helpful Percent')
plt.ylabel('Star Rating')
plt.legend (['Average Star Rating'])
plt.savefig("figure/Star Rating of Helpful Percent.png",dpi =500 , bbox_inches ='tight')
plt.show ()


del star_rating , helpful_star_frequency


# ######################################################
### 2.7. Plot 7
 # ######################################################


def product_date_process (dataset , threshold):
    product = dataset.groupby('product_id')['star_rating']. agg ({'count':'count'})
    product = product[product['count']> threshold ]. reset_index ()['product_id']

    product_date = dataset.groupby (['product_id','year','month'])['star_rating',' review_length']. agg ({'star_rating':['count','mean'],'review_length':'mean'})
    product_date .columns = ['count','avg_star','avg_length']
    product_date = product_date . reset_index ()
    product_date ['date'] = product_date ['year'] + product_date ['month'] / 12
    product_date = product_date .drop (['year','month'], axis =1).set_index (['product_id'
    ,'date']).loc[product]
    return product_date


product_date_h = product_date_process (new_hair_dryer , 400)
p, (ax1 , ax2 , ax3) = plt.subplots(nrows =3, ncols =1, sharex=True)
product_date_h ['count']. unstack ().T. sort_index ().plot(ax=ax1 , figsize =(5 ,7))
product_date_h ['avg_star']. unstack ().T. sort_index ().plot(ax=ax2 , legend=False)
product_date_h ['avg_length']. unstack ().T. sort_index ().plot(ax=ax3 , legend=False)
ax1. set_ylabel ('Review Amount')
ax2. set_ylabel ('Average Star Rating')
ax3. set_ylabel ('Average Review Length')
plt.savefig("figure/Product Infomation for Date of Hair Dryer.png",dpi =500 , bbox_inches
='tight')
plt.show ()

product_date_m = product_date_process (new_microwave , 78)
p, (ax1 , ax2 , ax3) = plt.subplots(nrows =3, ncols =1, sharex=True)
product_date_m ['count']. unstack ().T. sort_index ().plot(ax=ax1 , figsize =(5 ,7))
product_date_m ['avg_star']. unstack ().T. sort_index ().plot(ax=ax2 , legend=False)
product_date_m ['avg_length']. unstack ().T. sort_index ().plot(ax=ax3 , legend=False)
ax1. set_ylabel ('Review Amount')
ax2. set_ylabel ('Average Star Rating')
ax3. set_ylabel ('Average Review Length')
plt.xticks(np.arange (2012 , 2016))
plt.savefig("figure/Product Infomation for Date of Microwave .png",dpi =500 , bbox_inches ='tight')
plt.show ()

product_date_p = product_date_process (new_pacifier , 250)
p, (ax1 , ax2 , ax3) = plt.subplots(nrows =3, ncols =1, sharex=True)
product_date_p ['count']. unstack ().T. sort_index ().plot(ax=ax1 , figsize =(5 ,7))
product_date_p ['avg_star']. unstack ().T. sort_index ().plot(ax=ax2 , legend=False)
product_date_p ['avg_length']. unstack ().T. sort_index ().plot(ax=ax3 , legend=False)
ax1. set_ylabel ('Review Amount')
ax2. set_ylabel ('Average Star Rating')
ax3. set_ylabel ('Average Review Length')
plt.savefig("figure/Product Infomation for Date of Pacifier.png",dpi =500 , bbox_inches ='tight')
plt.show ()


product_date_h = product_date_process (new_hair_dryer , 100). reset_index ()
p, (ax1 , ax2 , ax3) = plt.subplots(nrows =1, ncols =3, sharey=True , figsize =(15 ,7))
ax1.scatter( product_date_h ['date'], product_date_h ['product_id'], product_date_h ['count']*10 , c='#ff0000', alpha =0.2)
ax2.scatter( product_date_h ['date'], product_date_h ['product_id'], product_date_h [' avg_star']*20 , c='#ff0000', alpha =0.2)
ax3.scatter( product_date_h ['date'], product_date_h ['product_id'], product_date_h [' avg_length']/5, c='#ff0000', alpha =0.2)
ax1.set_title ('Review Amount')
ax2.set_title ('Average Star Rating')
ax3.set_title ('Average Review Length')
ax1. set_xlabel ('Date')
ax2. set_xlabel ('Date')
ax3. set_xlabel ('Date')
ax1. set_ylabel ('Product ID')
plt.savefig("figure/Product Contrast for Date of Hair Dryer.png",dpi =100 , bbox_inches ='tight')
plt.show ()

product_date_m = product_date_process (new_microwave , 15). reset_index ()
p, (ax1 , ax2 , ax3) = plt.subplots(nrows =1, ncols =3, sharey=True , figsize =(15 ,7))
ax1.scatter( product_date_m ['date'], product_date_m ['product_id'], product_date_m ['count']*30 , c='#ff0000', alpha =0.2)
ax2.scatter( product_date_m ['date'], product_date_m ['product_id'], product_date_m [' avg_star']*20 , c='#ff0000', alpha =0.2)
ax3.scatter( product_date_m ['date'], product_date_m ['product_id'], product_date_m [' avg_length']/5, c='#ff0000', alpha =0.2)
ax1.set_title ('Review Amount')
ax2.set_title ('Average Star Rating')
ax3.set_title ('Average Review Length')
ax1. set_xlabel ('Date')
ax2. set_xlabel ('Date')
ax3. set_xlabel ('Date')
ax1. set_ylabel ('Product ID')
plt.savefig("figure/Product Contrast for Date of Microwave .png",dpi =100 , bbox_inches =' tight')
plt.show ()


product_date_p = product_date_process (new_pacifier , 60). reset_index ()
p, (ax1 , ax2 , ax3) = plt.subplots(nrows =1, ncols =3, sharey=True , figsize =(15 ,7))
ax1.scatter( product_date_p ['date'], product_date_p ['product_id'], product_date_p ['count']*10 , c='#ff0000', alpha =0.2)
ax2.scatter( product_date_p ['date'], product_date_p ['product_id'], product_date_p [' avg_star']*20 , c='#ff0000', alpha =0.2)
ax3.scatter( product_date_p ['date'], product_date_p ['product_id'], product_date_p [' avg_length']/5, c='#ff0000', alpha =0.2)
ax1.set_title ('Review Amount')
ax2.set_title ('Average Star Rating')
ax3.set_title ('Average Review Length')
ax1. set_xlabel ('Date')
ax2. set_xlabel ('Date')
ax3. set_xlabel ('Date')
ax1. set_ylabel ('Product ID')
plt.savefig("figure/Product Contrast for Date of Pacifier.png",dpi =100 , bbox_inches =' tight')
plt.show ()
del p, ax1 , ax2 , ax3 , product_date_h , product_date_m , product_date_p