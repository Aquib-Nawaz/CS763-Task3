import dlib
import cv2
import sys
import os
import numpy as np

# Load the XML file containing the facial landmark detector

if __name__ == '__main__':
    # Load the image dataset and its annotations
    pathtoxml = sys.argv[1]
    annotations = dlib.image_dataset_metadata.load_image_dataset_metadata(pathtoxml+os.sep+'landmarks.xml')
    red =  int(input("Enter Red"))
    green =  int(input("Enter Green"))
    blue =  int(input("Enter Blue"))
    synthesis_path = './data/synthesis/'
    for files in annotations.images:
        img = cv2.imread(pathtoxml+os.sep+files.filename)
        lines = []
        for box in files.boxes:
            mask = []
            for k, v in box.parts.items():
                mask.append([v.x,v.y])

            mask = mask[:14]+mask[-1:13:-1]
            mask = np.array(mask, np.int32)
            lines.append(mask)
        img2 = cv2.polylines(img, lines, True, (red,green,blue), thickness=2, lineType=cv2.LINE_8)
        img3 = cv2.fillPoly(img2, lines, (red,green,blue), lineType=cv2.LINE_AA)
        filedirec = os.path.dirname(files.filename)
        dirpath = synthesis_path+os.sep+filedirec
        try:
            os.makedirs(dirpath)
        except:
            pass
        cv2.imwrite(os.path.join(synthesis_path, files.filename),img3)
        # input("Enter any key")
    dlib.image_dataset_metadata.save_image_dataset_metadata(annotations, os.path.join(synthesis_path,'landmarks.xml'))
