from preprocess import *
from load_data import *
from evaluate_model import *
from prepare_text import *
# use vgg16 to convert images to features and save them
# directory='P:\\dataset\\Flicker8k_Dataset\\Images'
# features=extract_features(directory)
# print("extracted features ",len(features))
# dump(features,open('features.pkl','wb'))

#load and save descriptions

filename='P:\\dataset\\Flicker8k_Dataset\\Flickr8k.token.txt'

doc=load_doc(filename)
descriptions=load_description(doc)
clean_descriptions(descriptions)
vocabulary=to_vocabulary(descriptions)
print("vocabulary len: ", len(vocabulary))
save_description(descriptions,"description.txt")


filename='P:\dataset\Flicker8k_Dataset\Flickr_8k.trainImages.txt'
train_set=load_set(filename)
train_desc=load_clean_description("description.txt",train_set)
train_features=load_photo_features("features.pkl",train_set)
print("train set %d  train description %d  train_photo_features %d" % (len(train_set) ,len(train_desc) ,len(train_features)))
tokenizer=create_tokenizer(train_desc)
dump(tokenizer,open("tokenizer.pkl","wb"))

vocab_size=len(tokenizer.word_index)+1
max_length=max_length(descriptions=train_desc)
print(vocab_size)

x1train,x2train,ytrain=create_sequences(tokenizer,max_length,train_desc,vocab_size,train_features)

filename='P:\dataset\Flicker8k_Dataset\Flickr_8k.devImages.txt'
dev_set=load_set(filename)
dev_desc=load_clean_description("description.txt",dev_set)
dev_features=load_photo_features("features.pkl",dev_set)
x1dev,x2dev,ydev=create_sequences(tokenizer,max_length,dev_desc,vocab_size,dev_features)
print("dev set %d  dev description %d  dev_photo_features %d" % (len(dev_set) ,len(dev_desc) ,len(dev_features)))


model=define_model(vocab_size,max_length)
checkpoint=ModelCheckpoint('model.h5',monitor='val_loss',verbose=1,save_best_only=True,mode='min')
x1train=np.asarray(x1train).astype(np.float32)
x2train=np.asarray(x2train).astype(np.float32)

x1dev=np.asarray(x1dev).astype(np.float32)
x2dev=np.asarray(x2dev).astype(np.float32)

model.fit([x1train,x2train],ytrain,batch_size=16,epochs=5,verbose=2,callbacks=[checkpoint],validation_data=([x1dev,x2dev],ydev))






model=load_model("model.h5")
filename="P:\\dataset\\Flicker8k_Dataset\\Flickr_8k.testImages.txt"
test_set=load_set(filename)
tokenizer=load(open("tokenizer.pkl",'rb'))
test_descriptions=load_clean_description("description.txt",test_set)
test_features=load_photo_features("features.pkl",test_set)
evaluat_model(model,test_descriptions,test_features,tokenizer,34)