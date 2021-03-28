import tensorflow as tf
from tensorflow.python.keras import backend as K
from pickle import dump
# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

def load_set(filename):
    doc=load_doc(filename)
    dataset=list()
    for line in doc.split('\n'):
        if len(line)<1:
            continue
        id=line.split('.')[0]
        dataset.append(id)

    return set(dataset)

def load_clean_description(filename,dataset):
    doc=load_doc(filename)
    description=dict()
    for line in doc.split('\n'):
        tokens=line.split()
        image_id, image_desc=tokens[0],tokens[1:]
        image_id=image_id.split('.')[0]
        if image_id in dataset:
            if image_id not in description:
                description[image_id]=list()
            desc='startseq '+' '.join(image_desc)+' endseq'
            description[image_id].append(desc)
    return description


from pickle import load

def load_photo_features(filename,dataset):
    all_features=load(open(filename,'rb'))
    features={k:all_features[k] for k in dataset}
    return features


def to_line(descriptions):
    all_desc=list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return  all_desc

from keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):
    all_desc=to_line(descriptions)
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer



from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
def create_sequences(tokenizer,max_length,descriptions,vocab_size,photos):
    x1,x2,y=list(),list(),list()

    for key,desc_list in descriptions.items():
        for desc in desc_list:
            seq=tokenizer.texts_to_sequences([desc])[0]
            for i in range(1,len(seq)):
                in_seq,out_seq=seq[:i],seq[i]
                in_seq=pad_sequences([in_seq],max_length)[0]
                out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
                x1.append(photos[key][0])
                x2.append(in_seq)
                y.append(out_seq)
    return np.array(x1),np.array(x2),np.array(y)

def max_length(descriptions):
    lines=to_line(descriptions)
    return max(len(d.split()) for d in lines)

from keras.models import Model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model


def define_model(vocab_size,max_length):
     input1=Input(shape=(4096,))
     fe1=Dropout(0.5)(input1)
     fe2=Dense(256,activation='relu')(fe1)

     input2=Input(shape=(max_length,))
     se1=Embedding(vocab_size,256,mask_zero=True,)(input2)
     se2=Dropout(0.5)(se1)
     se3=LSTM(256)(se2)

     decoder1=add([fe2,se3])
     decoder2=Dense(256,activation='relu')(decoder1)
     outputs=Dense(vocab_size,activation='softmax')(decoder2)

     model=Model(inputs=[input1,input2],outputs=outputs)
     model.compile(loss='categorical_crossentropy',optimizer='adam')
     model.summary()
     plot_model(model,to_file="model.png",show_shapes=True)

     return model



# filename='P:\dataset\Flicker8k_Dataset\Flickr_8k.trainImages.txt'
# train_set=load_set(filename)
# train_desc=load_clean_description("description.txt",train_set)
# train_features=load_photo_features("features.pkl",train_set)
# print("train set %d  train description %d  train_photo_features %d" % (len(train_set) ,len(train_desc) ,len(train_features)))
# tokenizer=create_tokenizer(train_desc)
# dump(tokenizer,open("tokenizer.pkl","wb"))
#
# vocab_size=len(tokenizer.word_index)+1
# max_length=max_length(descriptions=train_desc)
# print(vocab_size)
#
# x1train,x2train,ytrain=create_sequences(tokenizer,max_length,train_desc,vocab_size,train_features)
#
# filename='P:\dataset\Flicker8k_Dataset\Flickr_8k.devImages.txt'
# dev_set=load_set(filename)
# dev_desc=load_clean_description("description.txt",dev_set)
# dev_features=load_photo_features("features.pkl",dev_set)
# x1dev,x2dev,ydev=create_sequences(tokenizer,max_length,dev_desc,vocab_size,dev_features)
# print("dev set %d  dev description %d  dev_photo_features %d" % (len(dev_set) ,len(dev_desc) ,len(dev_features)))
#
#
# model=define_model(vocab_size,max_length)
# checkpoint=ModelCheckpoint('model.h5',monitor='val_loss',verbose=1,save_best_only=True,mode='min')
# x1train=np.asarray(x1train).astype(np.float32)
# x2train=np.asarray(x2train).astype(np.float32)
#
# x1dev=np.asarray(x1dev).astype(np.float32)
# x2dev=np.asarray(x2dev).astype(np.float32)
#
# # ytrain=np.reshape(ytrain,(ytrain.shape[0],1))
# # ydev=np.reshape(ydev,(ydev.shape[0],1))
# # ytrain=np.asarray(ytrain).astype(np.float32)
# # ydev=np.asarray(ydev).astype(np.float32)
# model.fit([x1train,x2train],ytrain,batch_size=16,epochs=20,verbose=2,callbacks=[checkpoint],validation_data=([x1dev,x2dev],ydev))