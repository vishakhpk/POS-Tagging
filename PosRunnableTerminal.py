import gzip
import numpy
import sklearn 
from sklearn import neighbors, linear_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

def getContext(word):
    if '-' in word and word[0] != '-':
    	context=-1
    elif word.isdigit() and len(word) == 4:
        context=-2
    elif word[0].isdigit():
        context=-3
    else:
        context=0
    return context

trainDataset='./train.txt.gz'
testDataset='./test.txt.gz'

trainWordsX=[]
trainWordsY=[]

word2number={}
wordCounter=1

label2number={}
labelCounter=1

prefix2number={}
prefixCounter=1

combn2number={}
combnCounter=1
trainXCombn=[]

combn2number2={}
combnCounter2=1
trainXCombn2=[]    

suffix2number={}
suffixCounter=1

trainXNumbers=[]
trainYNumbers=[]

trainXPrefix=[]
trainXSuffix=[]

trainXPrefixNumbers=[]
trainXSuffixNumbers=[]

trainXPrevNumbers=[]
trainXPrev2=[]

trainXContext=[]
trainXContextPrev=[]

prev=0
prev2=0

print("Preparing Train Data....")
newContext=0
prevContext=0

with gzip.open(trainDataset,'rb') as f:
	for line in f:
	    list=line.split(' ')
	    #print(list)
	    if len(list)<2:
		#prev=0
		#prevContext=0
		continue
	    if list[1]=='.':
		#prev=0
		#prevContext=0
		continue
	    trainWordsX.append(list[0])
	    trainWordsY.append(list[1])
	    trainXPrevNumbers.append(prev)
	    trainXPrev2.append(prev2)
            tempCombn=prev2, prev
            prev2=prev
            #print tempCombn
            if not combn2number.has_key(tempCombn):
                combn2number[tempCombn]=combnCounter
                combnCounter=combnCounter+1
            trainXCombn.append(combn2number[tempCombn])
	    if not word2number.has_key(list[0]):
		word2number[list[0]]=wordCounter
		wordCounter=wordCounter+1
	    trainXNumbers.append(word2number[list[0]])
	    if not label2number.has_key(list[1]):
		label2number[list[1]]=labelCounter
		labelCounter=labelCounter+1
	    trainYNumbers.append(label2number[list[1]])
	    prev=label2number[list[1]]
	    trainXPrefix.append(list[0][0])
	    trainXSuffix.append(list[0][-3:])	    
	    if not prefix2number.has_key(list[0][0]):
		prefix2number[list[0][0]]=prefixCounter
		prefixCounter=prefixCounter+1
	    trainXPrefixNumbers.append(prefix2number[list[0][0]])
	    if not suffix2number.has_key(list[0][-3:]):
		suffix2number[list[0][-3:]]=suffixCounter
		suffixCounter=suffixCounter+1
	    trainXSuffixNumbers.append(suffix2number[list[0][-3:]])
	    newContext=getContext(list[0])
	    if newContext==0:
	    	newContext=word2number[list[0]]
	    tempCombn2=prev, newContext
            if not combn2number2.has_key(tempCombn2):
                combn2number2[tempCombn2]=combnCounter2
                combnCounter2=combnCounter2+1
            trainXCombn2.append(combn2number2[tempCombn2])
	    trainXContext.append(newContext)
	    trainXContextPrev.append(prevContext)
	    prevContext=newContext

trainXContextNext=[]
for i in xrange(0, len(trainXContext)-1):
	trainXContextNext.append(trainXContext[i+1])

trainXContextNext.append(0)

trainXContextNext2=[]
for i in xrange(0, len(trainXContext)-1):
	trainXContextNext2.append(trainXContextNext[i+1])

trainXContextNext2.append(0)

trainX=trainXCombn2, trainXPrevNumbers, trainXPrev2, trainXCombn, trainXPrefixNumbers, trainXSuffixNumbers, trainXContext, trainXContextPrev  #trainXContextNext, trainXContextNext2,
print(len(trainXNumbers), len(trainXPrevNumbers), len(trainXPrefixNumbers), len(trainXSuffixNumbers), len(trainXContext), len(trainXContextPrev), len(trainXContextNext), len(trainXContextNext2))
trainX=numpy.asarray(trainX)
trainX=numpy.transpose(trainX)
print(trainX.shape)
trainY=numpy.asarray(trainYNumbers)
trainY=trainY.reshape(len(trainY),1)
trainY=np_utils.to_categorical(trainY, 44)
#knn = neighbors.KNeighborsClassifier()
#logistic = linear_model.LogisticRegression()
#print('KNN Match score: %f' % knn.fit(trainX, trainY).score(trainX, trainY))
#print("Training SGD....")
#sgd=linear_model.SGDClassifier()
#print("Stochastic Gradient Descent: ", sgd.fit(trainX,trainY).score(trainX, trainY))
#print("Training Logistic....")
#print('LogisticRegression score: %f' % logistic.fit(trainX, trainY).score(trainX, trainY))
model=Sequential()
model.add(Dense(500,input_shape=(8,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(44))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(trainX[0:180000], trainY[0:180000],batch_size=500, nb_epoch=300, validation_data=(trainX[180000:190000], trainY[180000:190000]), verbose=1)
print("Evaluating:")
score=model.evaluate(trainX[190000:200000], trainY[190000:200000], verbose=0)
print "\nTest Score :",score[0], "\nTest Accuracy :", score[1]
"""
200k 50 epochs 2 layers 8 features 300 batch size
val_acc: 0.5937
Evaluating:

Test Score : 1.19356891595 
Test Accuracy : 0.60184

200k 50 epochs 2 layers 8 features 1000 batch size
val_loss: 1.2393 - val_acc: 0.5873
Evaluating:

Test Score : 1.21180478698 
Test Accuracy : 0.58988

so batch size doesn't make that much diff
let's try with 300 for now

200k 50 epochs 2 layers 10 features 300 batch size
val_loss: 1.5808 - val_acc: 0.4709
Evaluating:

Test Score : 1.53617566623 
Test Accuracy : 0.4882

so basically 10 features is not helpful so drop

200k 100 epochs 2 layers 8 features 300 batch size
val_loss: 1.1414 - val_acc: 0.6138
Evaluating:

Test Score : 1.10273591327 
Test Accuracy : 0.62668

extra 50 epochs 2% increase only :|
i want to see the effect of dropout so i'm just removing that for one run

200k 50 epochs 2 layers 8 features 300 batch size no dropout
val_loss: 1.2027 - val_acc: 0.5920
Evaluating:

Test Score : 1.17777849266 
Test Accuracy : 0.6006

dropout seems to have no difference at all
which makes sense if you think about it
how would you even generalise 200k words
probably it would have had a difference if smaller sample space

200k 50 epochs 3 layers 8 features 1000 batch size no dropout
val_loss: 1.1103 - val_acc: 0.6260
Evaluating:

Test Score : 1.08693324614 
Test Accuracy : 0.6328

200k 200 epochs 3 layers 8 features 500 batch size with dropout
val_loss: 0.8366 - val_acc: 0.7225
Evaluating:

Test Score : 0.825532695436 
Test Accuracy : 0.7216

200k 300 epochs 3 layers 8 features 500 batch size with dropout
val_loss: 0.8108 - val_acc: 0.8529
Evaluating:

Test Score : 0.69751461482 
Test Accuracy : 0.827

"""
