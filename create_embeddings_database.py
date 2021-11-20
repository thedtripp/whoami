from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
from numpy import expand_dims
import tensorflow as tf
from keras_vggface.utils import preprocess_input
import sys

model='quant_model'
modelpath='./' #if you have saved the model in a subfolder, add this folder path here
modelpath= modelpath + model + '.tflite'
PFAD = Path(Path.cwd() / 'All_croped_images/')
# PFAD = Path(Path.cwd() / 'test_croped_images/')
CelebFolders = next(os.walk(PFAD))[1]
EMBEDDINGS = pd.DataFrame()
ce=0
np.set_printoptions(threshold=sys.maxsize)# is needed to avoid ellipsis

# Load TFLite model and allocate tensors.Beide modelle funktionieren
#Depending on the version of TF running, check where lite is set :
print(tf.__version__)
if tf.__version__.startswith ('1.'):
    print('lite in dir(tf.contrib)' + str('lite' in dir(tf.contrib)))

elif tf.__version__.startswith ('2.'):
    print('lite in dir(tf)? ' + str('lite' in dir(tf)))
   
try: 
    interpreter = tf.lite.Interpreter(str(modelpath))   # input()    # To let the user see the error message
except ValueError as e:
    print("Error: Modelfile could not be found. Check if you are in the correct workdirectory. Errormessage:  " + str(e))
    import sys
    sys.exit()

#prepare the tflite model
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for celeb in CelebFolders:
    n= 0 #just for printing
    ce += 1 #just for printing
    print('-------------')
    print(str(celeb) + ' ' + str(ce) +' of '+str(len(CelebFolders))+ ' (' +str(ce*100/len(CelebFolders))+'%)')
    print('')
    filelist = next(os.walk(Path(PFAD / celeb)))[2] # go through each folder
    for f in filelist:
        n += 1 #just for printing
        try:
            img = cv2.imread(str(Path(PFAD / celeb / f)), cv2.IMREAD_COLOR).astype('float32')

            # Make images the same as they were trained on in the VGGface2 Model
            # convert one face into samples
            img = expand_dims(img, axis=0)#part of preprocessing
            img = preprocess_input(img, version=2)#part of preprocessing
            interpreter.set_tensor(input_details[0]['index'], img) # allocate tensor
            interpreter.invoke()        
            features = np.ravel(interpreter.get_tensor(output_details[0]['index'])) #calculate embeddings
            #now collect all the embeddings with the filenames and celeb names
            if EMBEDDINGS.empty:
                EMBEDDINGS = EMBEDDINGS.append({
                    'Name': celeb, 
                    'File': f, 
                    'Embedding': features
                            },
                    ignore_index=True,
                    sort=False)                
                Only_embeddings =list([features])
                Only_name = list([celeb])
                Only_file = list([f])
            else:
                EMBEDDINGS = EMBEDDINGS.append(
                    {
                        'Name': celeb,
                        'File': f,
                        'Embedding': features
                    }, 
                    ignore_index=True,
                    sort=False)
                Only_embeddings.append(features)
                Only_name.append(celeb)
                Only_file.append(f)
            if n==1:
                print('finished ' + str(n) + ' of ' + str(len(filelist)))
            else:
                print('         ' + str(n) + ' of ' + str(len(filelist)))
        except:
            continue   
filename_csv='EMBEDDINGS_' + model + '.csv'
filename_json='EMBEDDINGS_' + model + '.json'
EMBEDDINGS.to_csv(Path(Path.cwd() / filename_csv), index=False)
EMBEDDINGS.to_json(Path(Path.cwd() / filename_json))