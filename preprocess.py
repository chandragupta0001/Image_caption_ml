from os import listdir
from os import path
from keras.preprocessing.image import load_img ,img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory):
    images=dict()
    for name in listdir(directory):
        filename=path.join(directory,name)
        image = load_img(filename,target_size=(224,224))
        image=img_to_array(image)
        image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image=preprocess_input(image)
        image_id=name.split('.')[0]
        images[image_id]=image
    return images



# directory='P:\\dataset\\Flicker8k_Dataset\\Images'
# images=load_photos(directory)
# print("Loaded Images : %d" %len(images))

from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from pickle import dump
def extract_features(directory):
      in_layer=Input(shape=(224,224,3))
      model=VGG16()
      model.layers.pop()
      model=Model(inputs=model.inputs,outputs=model.layers[-1].output)
      features=dict()
      for name in listdir(directory):
          filename=path.join(directory,name)
          image=load_img(filename,target_size=(224,224))
          image =  img_to_array(image)
          image= image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
          image=preprocess_input(image)
          feature=model.predict(image,verbose=0)
          image_id=name.split('.')[0]

          features[image_id]=feature
          print('>',name)
      return features

# features=extract_features(directory)
# print("extracted features ",len(features))
# dump(features,open('features.pkl','wb'))

def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text


def laod_description(doc):
    mapping=dict()
    for line in doc.split('\n'):
        if len(line)<2:
            continue
        tokens=line.split()
        image_id , image_desc= tokens[0],tokens[1:]
        image_id=image_id.split('.')[0]
        image_desc=' '.join(image_desc)

        if image_id not in mapping:
            mapping[image_id]=image_desc
    return mapping


import re,string

def clean_description(description):
    re_punc=re.compile('[%s]'% re.escape(string.punctuation))

    for key,desc in description.items():
        desc=desc.split()
        desc =[word.lower() for word in desc]
        desc=[re_punc.sub('',w) for w in desc]
        desc = [word for word in desc if len(word)>1]
        description[key]=' '.join(desc)

def save_doc(description,filename):
    lines=list()
    for key, desc in description.items():
        lines.append(key + " "+ desc)
    data="\n".join(lines)
    file=open(filename,'w')
    file.write(data)
    file.close()

# filename="P:\\dataset\\Flicker8k_Dataset\\captions.txt"
# doc=load_doc(filename)
# description=laod_description(doc)
# print("loaded description ",len(description))
# clean_description(description)
# all_tokens=' '.join(description.values()).split()
# vocabulary=set(all_tokens)
#
# print('Vocabulary Size : %d' % len(vocabulary))
#
# save_doc(description,'description.txt')