
"""
file: celeb_face_matcher.py
title: Celebrity Face Mathcher
This is a module to be called from the web application.
It calculates an embedding from the captured face image
and finds the celebrity image with the minimum euclidean
distance. This should be the most similar looking celebrity.
"""
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os
from pathlib import Path
from time import time
import concurrent.futures
import sys

print('#Define functions')
#-----------------------------------------------------------------------------
def preprocess_input(x, data_format, version): #Choose version same as in " 2-Create embeddings database.py or jupyter"
    x_temp = np.copy(x)
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    # if version == 1:
    if data_format == 'channels_first':
        x_temp = x_temp[:, ::-1, ...]
        x_temp[:, 0, :, :] -= 93.5940
        x_temp[:, 1, :, :] -= 104.7624
        x_temp[:, 2, :, :] -= 129.1863
    else:
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= 93.5940
        x_temp[..., 1] -= 104.7624
        x_temp[..., 2] -= 129.1863

    return x_temp

def splitDataFrameIntoSmaller(df, chunkSize):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

def faceembeddingNP(YourFace,CelebDaten):
    Dist=[]
    for i in range(len(CelebDaten)):
        Celebs=np.array(CelebDaten[i]) 
        Dist.append(np.linalg.norm(YourFace-Celebs))
    return Dist

def white_rectangle(height):
    np_rect = np.zeros(height*448*3)
    np_rect = np_rect + 255
    np_rect = np.reshape(np_rect, (height, 448, 3))
    return np_rect

print('functions defined         ...')

def main(MY_FACE):

    print('Tensorflowversion: ' + tf.__version__)
    print('You work here: ', Path.cwd())

    print('# Define Variables')
    #-----------------------------------------------------------------------------
    Loadtype='TL'
    inputtype='float'
    largeImg=True

    ImgSize=(500,500) #Size of the plotted image IF largeImg IS SET TRUE
    Faces= False  # either True for only croped celebrity faces or False for original celbrity image. Deciding which images to show. Cropped or total resized image
    width=height=224 # size of the cropped image. Same as required for network

    modelpath='./'#'models/'
    embeddingpath='./'#'data/embeddings/'

    print('#TFLITE int8 QUANTIZED MODELS ')
    #*******************

    inputtype='float'
    embeddingsfile = 'EMBEDDINGS_quant_model.json'
    model='quant_model.tflite'
    model_path=str(Path.cwd() / modelpath / model)
        
    # Load Model tflite
    #-----------------------------------------------------------------------------
    print('# Load TFLite model and allocate tensors.')
    warm=time()
    try:  
        interpreter = tflite.Interpreter(model_path)
    except ValueError as e:
        print("Error: Modelfile could not be found. Check if you are in the correct workdirectory. Errormessage:  " + str(e))
        #Depending on the version of TF running, check where lite is set :
        if tf.__version__.startswith ('1.'):
            print('lite in dir(tf.contrib)' + str('lite' in dir(tf.contrib)))

        elif tf.__version__.startswith ('2.'):
            print('lite in dir(tf)? ' + str('lite' in dir(tf)))
        print('workdir: ' + os.getcwd())
        sys.exit()

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details() 

    print('model loaded with tflite...', time()-warm)

    print('# Load Embeddings')
    #-----------------------------------------------------------------------------
    emb=time()
    import json 

    f = open((Path.cwd() / embeddingpath  / embeddingsfile),'r') 
    ImportedData =json.load(f)
    dataE=[np.array(ImportedData['Embedding'][str(i)]) for i in range(len(ImportedData['Name']))]
    dataN=[np.array(ImportedData['Name'][str(i)]) for i in range(len(ImportedData['Name']))]
    dataF=[np.array(ImportedData['File'][str(i)]) for i in range(len(ImportedData['Name']))]

    print('Embeddings loaded      ...',time()-emb)

    print('# Split data for threadding')
    #-----------------------------------------------------------------------------
    splitt=time()
    celeb_embeddings=splitDataFrameIntoSmaller(dataE, int(np.ceil(len(dataE)/4)))   
    print('Embeddings split             ...' ,time()-splitt)

    start1 = time()

    # print('detect middle face ' ,time()-start1)
    # # FRAME THE DETECTED FACE
    start2=time()
    img=MY_FACE
    if len(img) != 0: # Check if face is out of the frame, then img=[], throwing error
        print('detect face ',time()-start2)

# CROP IMAGE 
        start3=time()
        if img.shape > (width,height): #downsampling
            img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) #resize the image to desired dimensions e.g., 224x224  
        elif img.shape < (width,height): #upsampling
            img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC) #resize the image to desired dimensions e.g., 224x224                      
        end3=time()
        print('face crop', end3-start3)
# IMAGE PREPROCESSING
        start4=time()
        if inputtype=='int':
            samples = np.expand_dims(img_small, axis=0)
            samples = preprocess_input(samples, data_format=None, version=3).astype('int8')#data_format= None, 'channels_last', 'channels_first' . If None, it is determined automatically from the backend
        else:
            pixels = img_small.astype('float32')
            samples = np.expand_dims( pixels, axis=0)
            samples = preprocess_input(samples, data_format=None, version=2)#data_format= None, 'channels_last', 'channels_first' . If None, it is determined automatically from the backend
        #now using the tflight model
        print('preprocess data for model' , time()-start4)
# CREATE FACE EMBEDDINGS                
        if Loadtype=='TL':
            prep=time()
            input_data = samples
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            print('ANN preperation ',time()-prep)
            start_e=time()
            EMBEDDINGS = interpreter.get_tensor(output_details[0]['index'])
        print('create face embeddings' , time()-start_e)
# READ CELEB EMBEDDINGS AND COMPARE  
        start_EU=time()
        EuDist=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result_1=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[0]))
            result_2=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[1]))
            result_3=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[2]))
            result_4=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[3]))

        if result_1.done() & result_2.done() & result_3.done() & result_4.done():
            EuDist.extend(result_1.result())
            EuDist.extend(result_2.result())
            EuDist.extend(result_3.result())
            EuDist.extend(result_4.result())
        print('Create_EuDist', time()-start_EU)

        start_Min=time()
        idx = np.argpartition(EuDist, 5)                
        folder_idx= dataN[np.argmin(EuDist)]
        image_idx = dataF[np.argmin(EuDist)] 
        print('find minimum for facematch', time()-start_Min)
        
# PLOT IMAGES       
        start6=time()

        pfad=str(Path.cwd() / 'All_croped_images' / str(folder_idx) / str(image_idx))


        # if Faces == False:
        #     pfad=str(Path.cwd() / 'All_croped_images' / str(folder_idx) / str(image_idx))
        # elif Faces == True:
        #     pfad=str(Path.cwd() / 'Celebs' / str(folder_idx) / str(image_idx))    
            
        Beleb=cv2.imread(pfad)
        if np.shape(Beleb) != (width,height): 
            Beleb=cv2.resize(Beleb, (np.shape(img_small)[0] ,np.shape(img_small)[1]), interpolation=cv2.INTER_AREA)
            
        if largeImg==True:
            larg=time()
            img_small2=cv2.resize(img_small, ImgSize, interpolation=cv2.INTER_LINEAR)
            Beleb2=cv2.resize(Beleb, (np.shape(img_small2)[0] ,np.shape(img_small2)[1]), interpolation=cv2.INTER_LINEAR)
            print('images upscaled ',time()-larg)
            numpy_horizontal2 = np.hstack((img_small2, Beleb2))

        numpy_horizontal = np.hstack((img_small, Beleb))

        rect_height_1 = 28
        rect_height_2 = 48
        np_rect_1 = white_rectangle(rect_height_1)
        np_rect_2 = white_rectangle(rect_height_2)

        numpy_horizontal = np.vstack((np_rect_1, numpy_horizontal, np_rect_2))

        font = 4 # FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
        org = (5, 253 + rect_height_1)
        fontScale = 0.9
        color = (0, 0, 0)
        thickness = 2
        celeb_name = str(dataN[np.argmin(EuDist)])
        numpy_horizontal = cv2.putText(numpy_horizontal, celeb_name, org, font, fontScale, color, thickness, cv2.LINE_AA) 
        
        print('print found image', time()-start6)
        print('-------------------------------------')
        print('time after face detection',time()-start1)
        print('-------------------------------------')                
        print('Distance value: ', EuDist[np.argmin(EuDist)].round(2), ' | ' , 'Name: ', str(dataN[np.argmin(EuDist)]),' | ' ,' Filename: ', str(dataF[np.argmin(EuDist)]))
        print('Top five celeb images: ')
        for i in range(5):
            print(dataN[idx[i]], 'Values: ',EuDist[idx[i]].round(2))
# CLEAR ALL VARIABLES
        faces_detected=None
        middle_face_X=None        
        img=None
        img_small=None
        pixels=None
        samples=None
        EMBEDDINGS=None          
        return numpy_horizontal

    else:
        print('FACE MUST BE IN FRAME')

if __name__ == "__main__":
    main()