#!/usr/bin/env/python
'''
SCORE FOR MEDR PREDICTIONS
---
Goal: Given medr_predictions.csv, compare all predictions against annotations medr_datagen_annotations.csv
and print accuracy score
---
'''

import pandas as pd
annodf = pd.read_csv('./medr_annotations.csv', dtype='object', encoding='latin-1')
#annodf.index.astype(str, copy=False)
annodf.rename(columns={
    '1':'RA1',
    '2':'RA2',
    '3':'RA3',
    '4':'RA4',
    '5':'RA5',
    '6':'RA6',
    '7':'RA7',
    '8':'RA8',
    '9':'RA9',
    '10':'RA10',
    '11':'RA11',
    '12':'RA12',
    '13':'RA13'
}, inplace=True)
annodf.fillna('NA', inplace=True) #missing values will be NA
print("Total Annotated ", annodf['id'].count())

preddf = pd.read_csv('./medr_predictions.csv', dtype='object', encoding='latin-1')
preddf.fillna('NA', inplace=True)
print("Total Predicted ", preddf['id'].count())

#preddf.index.astype(str, copy=False)

# merge
newdf = pd.merge(annodf, preddf, how='inner', on='id')
print("Total Predicted and Annotated", newdf['id'].count())
#print(newdf.head)
newdf_ups = newdf[newdf['survey'] == 'ups']
newdf_uob = newdf[newdf['survey'] == 'uob']
count_ups = newdf_ups['id'].count()
count_uob = newdf_uob['id'].count()
right_uob = []
right_ups = []
for i in range(13):
    right_uob.append(0)

for i in range(8):
    right_ups.append(0)

for q in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']:
    right = newdf_uob[newdf_uob['RA'+q] == newdf_uob['A'+q]]
    right_uob[int(q)-1]  = right['id'].count()

for q in ['1', '2', '3', '4', '5', '6', '7', '8']:
    right = newdf_ups[newdf_ups['RA'+q] == newdf_ups['A'+q]]
    right_ups[int(q)-1]  = right['id'].count()

right_uob_total = 0
right_ups_total = 0

print('UOB Total #of Surveys ', count_uob)
print('UOB #of Correct Answers per question')
for i in range(13):
    print('q', str(i+1), right_uob[i])
    right_uob_total += right_uob[i]
print('UOB % Correct Answers', "{:.2f}".format(100.0*right_uob_total/(13*count_uob)))
print('UPS Total #of Surveys ', count_ups)
print('UPS #of Correct Answers per question')
for i in range(8):
    print('q', str(i+1), right_ups[i])
    right_ups_total += right_ups[i]
print('UPS % Correct Answers', "{:.2f}".format(100.0*right_ups_total/(8*count_ups)))
print('Overall % Correct Answers ', "{:.2f}".format(100.0*(right_uob_total + right_ups_total)/(13*count_uob + 8*count_ups)))
