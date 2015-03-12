#__author__ = 'JChao'
import urllib2, time
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation


start_time=time.time()
#reading from specific url
url='http://www.mtsamples.com/site/pages/browse.asp?type=95-Radiology&page=1'
#read with BeautifulSoup
soup=BeautifulSoup(urllib2.urlopen(url).read())
url_list=[]
#when finding the url that links to a sample page, save it to list
for a in soup.find_all('a', href=True):
    if 'sample' in a['href']:
        url_list.append(a['href'])

#Although the two actions (above and below) can be combined into one loop, they are separated here in order to create
#simpler loops to make the code more readable

#print url_list


#for any single page
#search for the keywords


#need to create a megalist for classification. For each keyword that appears in each sample file that shows in megalist,
#set the value to 1, otherwise set it to 0 (supposedly, a mega_list should be already saved somewhere. This next step is
#only for the purpose of this project)

#create megalist
mega_list=[]
for link in url_list[:-1]:
    samp_url='http://www.mtsamples.com'+link
    samp_url = "%20".join( samp_url.split() )
    #print samp_url

    samp_soup=BeautifulSoup(urllib2.urlopen(samp_url).read())
    keywords=samp_soup.find('meta')['content']
    keywords=keywords.split(',')
    #print keywords
    for keys in keywords:
        if keys not in mega_list:
            mega_list.append(keys)


#creating tags for all cases, half of them will be used for training while the rest are for testing
#1) CT scam, 2)MRI 3) others
tag=[3,3,3,3,3,1,2,3,2,3,1,3,2,2,3,3,3,3,3,2,3,3,3,3,3,3,2,3,3,2,1,3,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,1,1,3,1,1,1,3,3,3,3,3,3,3,3,3,3,3,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,2,3,3,1,3,1,3,3,3,3,3,3,3,3,3,3,3,3,3,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

#The next step is to create a list for each report and then combine them into one 2D array
#273 = number of cases
feature_list=[[0]*len(mega_list)]*273
#create counters to use for index later
link_count=0
feature_count=0
#print len(mega_list)
for link in url_list[:-1]:

    #setup the url
    samp_url='http://www.mtsamples.com'+link
    samp_url = "%20".join( samp_url.split() )

    #soup it for keywords
    samp_soup=BeautifulSoup(urllib2.urlopen(samp_url).read())
    keywords=samp_soup.find('meta')['content']
    keywords=keywords.split(',')

    for keys in keywords:

        if keys in mega_list:
            feature_list[link_count][mega_list.index(keys)]=1


#The rest is classifying using the library scikit-learn (sklearn)
#cross validation for 10 times
result=[]
for i in range (1,11):
    feature_train, feature_test, tag_train, tag_test = cross_validation.train_test_split(feature_list, tag, test_size=0.5, random_state=0)
    clf = MultinomialNB()
    clf.fit(feature_train, tag_train)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    result.append(clf.score(feature_test, tag_test))

print "the average accuracy over 10 cross-validations is ", str(sum(result)/10*100), "%"
#checking the total time taken for this program
print "program took ", time.time()-start_time, "to run"
