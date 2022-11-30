import cv2 as cv # openCV 
import face_recognition # Face Recognition 
import numpy as np # numpy for linear algebra
import os # operating system for path
from datetime import datetime # date time for register attendance


direction=r'C:\Users\Kavinilavan\opencv\Face\train' 
images = [ ] # append images
labels = [ ] # append label names
List=os.listdir(direction) # it list out the each images in our folders
for i in List: # loop over the list
    framed_images = cv.imread(f'{direction}/{i}') # load the images using opencv
    images.append(framed_images)# append the images in the above list called images
    labels.append(os.path.splitext(i)[0])# append label names



# encoding the images
def create_encode(images):#function 
    encoded_images= [ ] # append encoded images
    for img in images: #loop over the images list which i created in above snipet 
        img=cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert BGR img to RGB
        encode=face_recognition.face_encodings(img)[0] # encode the image using face recognition lib
        encoded_images.append(encode) # append the enocded img in above list called encoded_images
    return encoded_images # return the function 

encode_known=create_encode(images) # function calling


# Attendence marking
def register_attnd(label): # function for attendence marking 
    with open(r"C:\Users\Kavinilavan\opencv\Face-Recog\Attend.csv",'r+') as f: # open and read the csv file
        myattend=f.readlines() # returns a list containing each line in the file as a list item.
        students=[ ] # append 
        for line in myattend: #loop over in the myattend
            entry=line.split(',') # entry the attendence name , time 
            students.append(entry[0])# append the entry in students 
        if label not in students: # if condition for dont repeat 
            now = datetime.now() # current time
            times = now.strftime('%H:%M:%S')# time frame
            f.writelines(f'\n{label},{times}') # writ name and time in csv





# web cam
cap = cv.VideoCapture(0)

while True: # infinity loop
    Frame, img = cap.read() # read image
    img_small= cv.resize(img,(0,0),None, 0.50,0.50)# resize the img because of efficiency
    img_small=cv.cvtColor(img_small, cv.COLOR_BGR2RGB)# convert into RGB
    FaceLocation=face_recognition.face_locations(img_small)# locate the face using face_recognition library
    EncodeCurrentFrame = face_recognition.face_encodings(img_small,FaceLocation)# encoding the face which we locate in above code
    for en,lo in zip(EncodeCurrentFrame,FaceLocation): # zip for working both variable in same operation
        compare=face_recognition.compare_faces(encode_known,en)# compare with all other training images
        dist=face_recognition.face_distance(encode_known, en)# calculate the distance
        print(dist)# print the distance
        compareind=np.argmin(dist)# compare index
        if compare[compareind]:# if comparing face is match , the below operation will be occured
            label= labels[compareind].upper()# convert the labels into upper case
            print(label)# print the labels name
            h1, l2, h2, l1 = lo# frame for bounding box to faces
            h1, l2, h2, l1 = h1*2, l2*2, h2*2, l1*2
            cv.rectangle(img,(l1,h1),(l2,h2),(255,0,0),4)# rectangle for faces
            cv.rectangle(img,(l1, h2-35),(l2,h2),(255,0,0),cv.FILLED) # color for rectangle
            cv.putText(img,label,(l1+6,h2-6),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),3) # text below the rectangle
            register_attnd(labels)


    cv.imshow("Frame",img)
    cv.waitKey(1)


