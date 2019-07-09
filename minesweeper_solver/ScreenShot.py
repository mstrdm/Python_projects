## Takes a screenshot of the Minesweeper window.
from pylab import *
import win32gui
import win32ui
from ctypes import windll
import Image

Minesweeper = win32gui.FindWindow(None, 'Minesweeper')

# Change the line below depending on whether you want the whole window
# or just the client area. 

left, top, right, bot = win32gui.GetWindowRect(Minesweeper)
w = right - left
h = bot - top

hwndDC = win32gui.GetWindowDC(Minesweeper)
mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()

saveBitMap = win32ui.CreateBitmap()
saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

saveDC.SelectObject(saveBitMap)

# Change the line below depending on whether you want the whole window
# or just the client area. 

result = windll.user32.PrintWindow(Minesweeper, saveDC.GetSafeHdc(), 0)
print result

bmpinfo = saveBitMap.GetInfo()
bmpstr = saveBitMap.GetBitmapBits(True)

im = Image.frombuffer(
    'RGB',
    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
    bmpstr, 'raw', 'BGRX', 0, 1)

win32gui.DeleteObject(saveBitMap.GetHandle())
saveDC.DeleteDC()
mfcDC.DeleteDC()
win32gui.ReleaseDC(Minesweeper, hwndDC)

if result == 1:
    #PrintWindow Succeeded
    im.save("ms.bmp")