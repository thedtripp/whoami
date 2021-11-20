import cv2
from pathlib import Path
import os

#Allocate face detection classifier
# face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

#set file path
path = Path.cwd() / 'Celebs/'  
savefolder = Path(Path.cwd() / 'All_croped_images/')
savefolder.mkdir(parents=True, exist_ok=True)

#set variables
p = 50  #Buffer for space around detected face to croping
width = 224
height = width

Folderlist = next(os.walk(path))[1] #get all folder names
#print(Folderlist)

for celeb in Folderlist: # now go throug all folders
    filelist = next(os.walk(Path(path / celeb)))[2]
    print(celeb)
    for f in filelist:  #Listing jpg files in this directory tree
        img = cv2.imread(str(Path(path / celeb / f)), cv2.IMREAD_COLOR) # read each image
        print(f)
        
        #Detect face
        faces_detected = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        if len(faces_detected) != 0:  # only if the cascader detected a face, otherwise error
            (x, y, w, h) = faces_detected[0] # coordinates of box around face
            #create folderstructure with a new folder for each celebrity
            croppedpath = Path(savefolder / celeb)
            os.makedirs(croppedpath, exist_ok=True) # differnt way than above to create folders
            filename = f'{croppedpath}/{f}'
            #Crop image to face
            img = img[y - p + 1:y + h + p, x - p + 1:x + w + p]  #use only the detected face; crop it
            if img.shape > (width, height) and img.size is not 0:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)  #resize the image to desired dimensions e.g., 256x256
                #Save croped image in folder
                cv2.imwrite(filename, img)  #save image in folder
            else:
                print('image to small or facebox out of image')
        else:
            print('no face detected')