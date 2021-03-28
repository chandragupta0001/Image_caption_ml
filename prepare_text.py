import re
import string
def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

def load_description(doc):
    mapping=dict()
    for line in doc.split('\n'):
        tokens=line.split()
        if len(tokens)<2:
            continue
        image_id,image_desc=tokens[0],tokens[1:]
        image_id=image_id.split('.')[0]
        image_desc=' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id]=list()
        mapping[image_id].append(image_desc)

    return mapping

def clean_descriptions(descriptions):
    re_punc=re.compile('[%s]' %re.escape((string.punctuation)))

    for key,desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc=desc_list[i]
            desc=desc.split()
            desc=[word.lower() for word in desc]
            desc=[re_punc.sub('',w) for w in desc]
            desc=[word for word in desc if len(word)>1 and word.isalpha()]
            desc_list[i]=' '.join(desc)

def to_vocabulary(descriptions):
    all_desc=set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def save_description(descriptions,filename):
    lines=list()

    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + " " + desc)

    data='\n'.join(lines)
    file=open(filename,'w')
    file.write(data)
    file.close()


# filename='P:\\dataset\\Flicker8k_Dataset\\Flickr8k.token.txt'
#
# doc=load_doc(filename)
# descriptions=load_description(doc)
# clean_descriptions(descriptions)
# vocabulary=to_vocabulary(descriptions)
# print("vocabulary len: ", len(vocabulary))
# save_description(descriptions,"description.txt")
