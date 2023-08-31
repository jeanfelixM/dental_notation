import cv2
from PIL import Image, ImageDraw
import copy
import numpy as np

def preprocess(img,n=5,s=11,debug=False):
    img = cv2.GaussianBlur(img,(n,n),s)
    if debug:
        print("Blured")
        saveImage(img, "testBlur.png")
    return img

def keepCircle(img,bary=True,complete=False,inf=600,sup=10000,debug=False):
    pixelLeft = set(zip(*np.where(img!=0)))
    barys = []
    
    
    while len(pixelLeft) > 0:
        zone = set()
        fpixel = pixelLeft.pop()
        stack = [fpixel]
        zone.add(fpixel)
        while stack:
            pixelco = stack.pop()
            for i in range(-1,2):
                for j in range(-1,2):
                    if (pixelco[0]+i,pixelco[1]+j) in pixelLeft:
                        zone.add((pixelco[0]+i,pixelco[1]+j))
                        stack.append((pixelco[0]+i,pixelco[1]+j))
                        pixelLeft.remove((pixelco[0]+i,pixelco[1]+j))
        if shouldDelete((len(zone)),inf,sup):
            img[list(zip(*zone))] = 0
        elif bary:
            indices = list(zip(*zone))
            barycenter = np.average(indices, axis=1, weights=img[indices])
            if debug:
                print("Barycentre:", barycenter)
            barys.append([barycenter[1],barycenter[0]])
    if debug:
        saveImage(img, "testCleaning.png")
    
    if complete:
        if len(barys) == 0:
            h, w = img.shape
            return img,[[w, h]]
    if bary:
        return img,barys
    else:
        return img
            
def shouldDelete(size,inf=600,sup=10000):
    return (size < inf or size > sup)

def saveImage(img, name):
    img = Image.fromarray(img)
    img.save(name)
    

def findCenter(img,complete=False,inf=600,sup=10000):
    img = preprocess(img)
    img,barys = keepCircle(img,complete,inf,sup)
    for barycenter in barys:
        cv2.circle(img, (int(barycenter[0]), int(barycenter[1])), radius=5, color=(255, 0, 0), thickness=-1)
    saveImage(img, "testBarycenter.png")
    return barys

    
def main():
    print("Start")
    img = cv2.imread("output.png",0)
    img = preprocess(img)
    img,barys = keepCircle(img,True)
    for barycenter in barys:
        cv2.circle(img, (int(barycenter[0]), int(barycenter[1])), radius=5, color=(255, 0, 0), thickness=-1)
    saveImage(img, "testBarycenter.png")
    #print("Barycentres:", barys)

if __name__ == "__main__":
    main()
