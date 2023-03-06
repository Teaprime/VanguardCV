import math
import time

import numpy as np
import cv2 as cv
import pyautogui as pag
import mouse
import keyboard
import pydirectinput
from PIL import Image, ImageGrab
from mss import mss
from WindowCapture import WindowCapture

import os


img = cv.imread('test.png',cv.IMREAD_COLOR)
img=cv.bitwise_not(img)
print(type(img))
img2 = img.copy()


rawX=0
rawY=0

prevTime=0

testvar=True
found=False

gameResX=1920
gameResY=1080
manualoffsetx=0
manualoffsety=int(0.05*gameResY)

pydirectinput.FAILSAFE=False

template = cv.imread('marker3.png',cv.IMREAD_COLOR)
#cv.cvtColor(template,cv.COLOR_RGB2HSV,template)
#cv.imshow('marker',template)
#cv.resizeWindow('marker',200,100)
w, h = template.shape[::-2]
templateGpu=cv.UMat(template)
#cv.imshow("test",templateGpu)

cap=cv.VideoCapture(1)

def mouseSmooth(x,y):
    newX=10
    newY=10
    if abs(x)>1:
        newX=int(x/(math.log2(abs(x))/2))
    if abs(y) > 1:
        newY=int(y/(math.log2(abs(y))/2))
    if x>0:
        newX=-newX
    if y>0:
        newY=-newY
    return (newX,newY)

def evaluateFrame(img):
    #method=cv.TM_CCOEFF_NORMED
    #img2=cv.UMat(img)
    #cv.imshow("test",img2)
    method=cv.TM_CCOEFF_NORMED
    #method=cv.sq
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    coord_top_left = max_loc
    #coord_top_left = min_loc
    certainty=max_val
    #certainty=min_val
    coordsCenter= (coord_top_left[0]+8+525+manualoffsetx,coord_top_left[1]+12+280+manualoffsety)
    return (coordsCenter,certainty)

def matchcenter(coords):
    #print("todo")

    #rawX=960-(1920-coordstopLeft[0])+10
    rawX= -(gameResX/2 - coords[0])
    #rawY=540-(1080-coordstopLeft[1])+25
    rawY= -(gameResY/2 - coords[1])

    offsetX=int((rawX))
    offsetY=int((rawY))
    newTup=(offsetX,offsetY)
    return newTup

#img=cv.imread('test.png')


sct=mss()
mon=sct.monitors[0]
monitor={"top":mon["top"]+int(0.26*gameResY),"left":mon["left"]+int(0.27*gameResX),"width":int(0.45*gameResX),"height":int(0.45*gameResY)}
frame=None

wincap = WindowCapture()

fps=0
while True:
    start_time=time.time()
    # Capture frame-by-frame
    #cv.imshow('frame',img)


    #frame=pag.screenshot(region=(525,280,870,489))
    #pilGrab=ImageGrab.grab((525,280,525+870,280+489))
    #frame=np.array(pilGrab)

    #frame=pag.screenshot(region=(525,280,870,489))
    #frame=framePre[600:1200,300:600]
    #frame=np.array(frame)

    #frame=cv.UMat(np.array(sct.grab(monitor))[:,:,:3])

    _, frame = cap.read()
    #frame=cv.cvtColor(frame, cv.COLOR_B)
    print(frame.shape)

    #frame=np.array(sct.grab(monitor))[:,:,:3]
    #frame=cv.UMat(frame)

    #frame=sct.grab(monitor)
    #frame.reshape(1080,1920,3)

    #frame = wincap.get_screenshot()

    #frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    #frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    #frame=cv.cvtColor(frame,cv.COLOR_RGB2BGR)

    #frame=cv.bitwise_not(frame)

    #print(frame.shape)
    #"""
    top_left,certainty=evaluateFrame(frame)
    #print(certainty)
    if certainty>0.23:
        cv.circle(frame, (top_left[0]-int(0.27*gameResX),top_left[1]-int(0.26*gameResY)), 5, (0, 0, 255), -1)
        found=True
    else:
        found=False
    #"""
    cv.putText(frame,str(certainty),(10,10),0, 0.3, (255,255,255))
    cv.putText(frame,str(fps),(10,20),0, 0.3, (255,255,255))
    cv.imshow('frame', frame)
    #cv.resizeWindow('frame',frame.shape[1::-1])
    """
    try:
        if keyboard.is_pressed('b'):
            x,y=matchcenter(top_left)
            mouse.move(x,y,False,0.005)
        ""
        elif keyboard.is_pressed('q'):
            cv.destroyAllWindows()
            break;
        ""
    except:
        print()
    """

    #if keyboard.is_pressed('b'):
    if keyboard.is_pressed('c') and found:
        """
        xDone=False
        yDone=False
        tick=0
        while (tick<200) or (xDone and yDone):
            x, y = matchcenter(top_left)
            if not abs(matchcenter(top_left)[0])<4:
                if x<0:
                    pydirectinput.moveRel(-1,0)
                elif x>0:
                    pydirectinput.moveRel(1, 0)
            else:
                xDone=True
    
            if not abs(matchcenter(top_left)[1])<4:
                if x<0:
                    pydirectinput.moveRel(0,-1)
                elif x>0:
                    pydirectinput.moveRel(0, 1)
            else:
                xDone=True
            tick+=1
            """
        x, y = matchcenter(top_left)
        #y=math.sin((y/486)*math.pi)*486/10
        #x=math.sin((x/864)*math.pi)*864/10
        x=math.sinh((x/864)*math.pi)*100
        y=math.sin((y/486)*math.pi)*40
        print("sin value: "+str(x))
        #smoothed=mouseSmooth(x,y)
        """
        if (abs(rawX)>200) or (abs(rawY)>100):
            print("skipping")
        else:
            """
    #mouse.move(x, y, False, 0.01)

        pydirectinput.moveRel(int(x),int(y),0)
        #pydirectinput.moveRel(20,20,0)
        """
        if testvar:
            print("lmao")
        else:
            print("sock")
        testvar=not testvar
        """
    #pydirectinput.moveRel(smoothed[0],smoothed[1],None)
    #pydirectinput.move(rawX,rawY,0)
    fps="FPS: "+str( 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    #print(fps)  # FPS = 1 / time to process loop

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
# When everything done, release the capture
