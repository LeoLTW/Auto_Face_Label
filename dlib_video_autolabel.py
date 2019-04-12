
# coding: utf-8

# In[1]:


import cv2
from imutils.face_utils import rect_to_bb
import dlib
import imutils
import os, time

#-----------------------------------
datasetPath = "faceYolo_vide/"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

cam_resolution = (1280,1024)
labelName = "face"
minFaceSize = (30, 30)

dlib_detectorRatio = 1
folderCharacter = "/"  # \\ is for windows

video_folder = "media/sf_VMshare/videos"

#------------------------------------

if not os.path.exists(datasetPath):
    os.makedirs(datasetPath)

if not os.path.exists(datasetPath + imgPath):
    os.makedirs(datasetPath + imgPath)

if not os.path.exists(datasetPath + labelPath):
    os.makedirs(datasetPath + labelPath)

if not os.path.exists('media/sf_VMshare/videos'):
    os.makedirs('media/sf_VMshare/videos')


# In[2]:


def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def getFaces_dlib(img):
    detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    bboxes = []
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            bboxes.append((x, y, w, h))

    return bboxes


def generateXML(img, filename, fullpath, bboxes):
    xmlObject = ""
    for bbox in bboxes:
        xmlObject = xmlObject + writeObjects(labelName, bbox)

    print("SHAPE:", img.shape)
    with open(xml_file) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image


# In[3]:


start_time = time.time()
width = 800
height = 800

i = 0

for file in os.listdir(video_folder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    noFrame = False
    print(video_folder + "/" + file)
    if(file_extension==".mp4" or file_extension==".mov" or file_extension==".mk4" or file_extension==".avi"):
        VIDEO_IN = cv2.VideoCapture(video_folder + "/" + file)

        while noFrame is False:
            hasFrame, frame = VIDEO_IN.read()
            if not hasFrame:
                print("Done processing !!!")
                print("--- %s seconds ---" % (time.time() - start_time))
                noFrame = True
                break

            #frame = imutils.rotate(frame, 90)
            
            faces = getFaces_dlib(frame)

            frameCopy = frame.copy()
            #frameCopy = putText(frameCopy, str(len(faces))+" faces", 10,30, color=(0,255,0), thickness=2, size=0.8)

            if(len(faces)>0):
                for (x,y,w,h) in faces:
                    cv2.rectangle( frameCopy,(x,y),(x+w,y+h),(0,255,0),2)

                filename = "face_" + str(time.time()) + "." + str(i)
                #save images to dataset
                cv2.imwrite(datasetPath + imgPath + filename + "." + imgType, frame)

                xmlfilename = filename + ".xml"
                xmlContent = generateXML(frame, xmlfilename, datasetPath + labelPath + xmlfilename, faces)
                file = open(datasetPath + labelPath + xmlfilename, "w")
                file.write(xmlContent)
                file.close

                #frameCopy = putText(frameCopy, "saved to "+xmlfilename, 10,80, color=(0,255,0), thickness=2, size=0.8)
                #out.write(frameCopy)
                print("Frame {} processed, {} faces.".format(i, len(faces)))

            else:
                #out.write(frameCopy)
                print("Frame {} processed, no face.".format(i))

            i += 1
            #cv2.imshow("Frame", imutils.resize(frameCopy, width=640))
            #cv2.waitKey(1)

