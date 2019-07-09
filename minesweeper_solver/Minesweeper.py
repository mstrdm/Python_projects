## Minesweeper solver.
## Controls the mouse to solve Minesweeper game. Requires Windows 7 type game appearance.
## Solver doesn't use difficult guessing or estimation techinques, so it doesn't work well on highest difficulty.

import win32api
import win32con
from pylab import *
import numpy as np
import cv2
import win32gui
import win32ui
from ctypes import windll
from PIL import Image
#import Image
import time

tot_x=16; tot_y=16 #Difficulty level

def flagSpace(x,y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
    time.sleep(0.01) #wait a bit for animations to finish ~0.01s
def openSpace(x,y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
    time.sleep(0.01) #wait a bit for animations to finish ~0.01s
def openAll(x,y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP,0,0)
    time.sleep(0.01) #wait a bit for animations to finish ~0.01s
    
###############################################################################
def getBoard():
    
    Board = zeros((tot_y,tot_x))
    printScreen()
    img_gray = cv2.cvtColor(cv2.imread('Board.bmp'), cv2.COLOR_BGR2GRAY)
    
    for spaceType in arange(1,11):
#        if spaceType == 9:
  
        template = cv2.imread(str(spaceType)+'.png',0)
        temp_w, temp_h = template.shape[::-1]
    
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        if spaceType == 9 or 11:
            threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            bx = int((pt[0]-37+7)/18); by = int((pt[1]-79+7)/18)
            Board[by,bx] = spaceType
            if Board[bx,by] == 9 or Board[bx,by] == 11:
                mines[bx,by] = 1; Board[bx,by] = 9
    time.sleep(0.01)
    return Board

###############################################################################
def printScreen():
    
    hwndDC = win32gui.GetWindowDC(Minesweeper)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    result = windll.user32.PrintWindow(Minesweeper, saveDC.GetSafeHdc(), 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer('RGB',(bmpinfo['bmWidth'], bmpinfo['bmHeight']),bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(Minesweeper, hwndDC)

    if result == 1:
        im.save("Board.bmp")
        
###############################################################################
def checkMatch(M,y,x): #checks if the number op closed spaces around the number is equal to the number and if yes, marks mines
    N=0
    for dhor in arange(-1,2):
        for dver in arange(-1,2):
            if x+dhor>=0 and y+dver>=0 and x+dhor<tot_x and y+dver<tot_y:
                if Board[y+dver,x+dhor]==10 or mines[y+dver,x+dhor]==1:
                    N=N+1
    if N==M:
        solved[y,x] = 1
        for dhor in arange(-1,2):
            for dver in arange(-1,2):
                if x+dhor>=0 and y+dver>=0 and x+dhor<tot_x and y+dver<tot_y:
                    if Board[y+dver,x+dhor]==10 and mines[y+dver,x+dhor]==0:
                        mines[y+dver,x+dhor] = 1
                        flagSpace(first_left+(x+dhor)*dx,first_top+(y+dver)*dx)
    time.sleep(0.01) #wait a bit for animations to finish ~0.01s

###############################################################################
def clean(M,y,x): #middle-clicks the number if all mines around that number have been marked
    N=0
    for dhor in arange(-1,2):
        for dver in arange(-1,2):
            if x+dhor>=0 and y+dver>=0 and x+dhor<tot_x and y+dver<tot_y:
                N = N + mines[y+dver,x+dhor]
    if N==M:
        openAll(first_left+x*dx,first_top+y*dx)
        solved[y,x] = 1
        
        for dhor in arange(-1,2):
            for dver in arange(-1,2):
                if x+dhor>=0 and y+dver>=0 and x+dhor<tot_x and y+dver<tot_y and mines[y+dver,x+dhor]!=1:
                    safe[y+dver,x+dhor] = 1  
    time.sleep(0.01) #wait a bit for animations to finish ~0.01s

###############################################################################
def checkMines():
    for y in arange(tot_y):
        for x in arange(tot_x):
            if mines[y,x]==1 and Board[y,x]!=9:
                flagSpace(first_left+x*dx,first_top+y*dx)
#    time.sleep(0.01) #wait a bit for animations to finish ~0.01s

###############################################################################
def checkSafe():
    for y in arange(tot_y):
        for x in arange(tot_x):
            if safe[y,x]==1 and Board[y,x]==10:
                openSpace(first_left+x*dx,first_top+y*dx)
                
###############################################################################                

Minesweeper = win32gui.FindWindow(None, 'Minesweeper')
left, top, right, bot = win32gui.GetWindowRect(Minesweeper)
w = right - left; h = bot - top
first_left = left+47; first_top = top+89 #first space 47;89
dx = 18 #pixels between spaces

openSpace(left,top)
openSpace(first_left+7*dx,first_top+7*dx) #Open the first space
time.sleep(0.15) #wait for the fancy animation to end ~0.15s

mines = zeros((tot_y,tot_x)) #array for checking mines so that program doesn't mark the same one two times
solved = zeros((tot_y,tot_x)) #array to mark the solved spaces to not go back to them again
safe = zeros((tot_y,tot_x)) #array to mark spaces with no mines to check if they are opened in the program later

while sum(mines)<40:
    Board = getBoard()    
    for y in arange(tot_y):
        for x in arange(tot_x):
            if Board[y,x]<9 and Board[y,x]>0 and solved[y,x]==0: #if selected space is a number
                checkMatch(Board[y,x],y,x)
    for y in arange(tot_y):
        for x in arange(tot_x):
            if Board[y,x]<9 and Board[y,x]>0 and solved[y,x]==0: #if selected space is a number
                clean(Board[y,x],y,x)    
    Board = getBoard()
    checkSafe()






























    
    
    
    
    
    
    
    
    
    