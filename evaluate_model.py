from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
from load_data import load_clean_description,load_set,load_photo_features,max_length
from PIL import Image
def word_for_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index== integer:
            return word
    return None

def gen_desc(model,tokenizer,photo,max_length):
    in_text="startseq"

    for i in range(max_length):
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        sequence=pad_sequences([sequence],maxlen=max_length)
        yhat=model.predict([photo,sequence],verbose=0)
        yhat=argmax(yhat)
        word=word_for_id(yhat,tokenizer)

        if word is None:
            break
        in_text= in_text+' '+word
        if word=='endseq':
            break

    return in_text


def clean_summary(summary):
    index=summary.find('startseq')
    if index>-1:
        summary=summary[len('startseq'):]
    index=summary.find('endseq')
    if index>-1:
        summary=summary[:index]
    return summary



def evaluat_model(model,descriptions,photos,tokenizer,max_length):
    actual,prediction=list(),list()
    for key,desc_list in descriptions.items():
        yhat=gen_desc(model,tokenizer,photos[key],max_length)
        yhat=clean_summary(yhat)
        references=[clean_summary(d).split() for d in desc_list]
        actual.append(references)
        prediction.append(yhat.split())
        # print(yhat)
        # image=Image.open("P:\\dataset\\Flicker8k_Dataset\\Images\\"+key+".jpg")
        # image.show()

    print('BLEU-1: %f' % corpus_bleu(actual, prediction, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, prediction, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, prediction, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, prediction, weights=(0.25, 0.25, 0.25, 0.25)))

# model=load_model("model.h5")
# filename="P:\\dataset\\Flicker8k_Dataset\\Flickr_8k.testImages.txt"
# test_set=load_set(filename)
# tokenizer=load(open("tokenizer.pkl",'rb'))
# test_descriptions=load_clean_description("description.txt",test_set)
# test_features=load_photo_features("features.pkl",test_set)
# evaluat_model(model,test_descriptions,test_features,tokenizer,34)
