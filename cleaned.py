import sys
import cv2 as cv
import numpy as np

####### this is the sample code from the given source.


def show_wait_destroy(winname, img):
    cv.imwrite(str(winname)+".png", img)
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


inp_sheet = cv.imread("data/sample.png", cv.IMREAD_COLOR)
corners1 = np.copy(inp_sheet)
inp = np.copy(inp_sheet)

if len(inp_sheet.shape) != 2:
    gray = cv.cvtColor(inp_sheet, cv.COLOR_BGR2GRAY)
else:
    gray = inp_sheet

show_wait_destroy("gray", gray)

# Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
gray = cv.bitwise_not(gray)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

# Create the images that will use to extract the horizontal lines
horizontal = np.copy(bw)
bw2 = np.copy(bw)

def _lines(horizontal,name):
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    #show_wait_destroy(name, horizontal)
    return horizontal

horizontal = _lines(horizontal,"lines")
# Remove horizontal

def remove_lines(b,inp):
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,1))
    detected_lines = cv.morphologyEx(b, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(inp_sheet, [c], -1, (255,255,255), 2)
    # Repair image
    repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,6))
    result = 255 - cv.morphologyEx(255 - inp, cv.MORPH_CLOSE, repair_kernel, iterations=1)
    show_wait_destroy("removed lines", result)
    return result


#detecting corners
def corners(pic,detect, name):
    blur = cv.GaussianBlur(detect, (3, 3), 0)
    thresh = cv.threshold(blur, 220, 255, cv.THRESH_BINARY_INV)[1]

    x, y, w, h = cv.boundingRect(thresh)

    me2 = (np.argmax(thresh[y, :]), y) #leftup
    me1 = (x + w - 1, y+h-1) #rightdown
    me4 = (x + w - 1, y) #rightupmost
    me5 = (x, np.argmax(thresh[:, x])) #leftdown
    cv.circle(pic, me2, 8, (255, 255, 0), -1)
    cv.circle(pic, me1, 8, (255, 255, 0), -1)
    cv.circle(pic, me4, 8, (255, 255, 0), -1)
    cv.circle(pic, me5, 8, (255, 255, 0), -1)

    return pic, me2, me4, me1, me5, x,y,w,h

def cut(pic, x,y,w,h, name):
    roi = pic[y:y + h, x:x + w]
    roi = cv.resize(roi, (w, h))
    show_wait_destroy(name, roi)
    return roi

lines = cv.bitwise_not(horizontal)
corners1, left, right, top, bottom, x, y, w, h1= corners(corners1,lines,"corners")
point_inp = np.array([left,right,top,bottom], dtype=np.float32)


#%%HOMOGRAPHY

H=inp.shape[0]
W=inp.shape[1]
dst = np.array([[0, 0],   [W, 0],   [W, H],    [0, H]], np.float32)
h, status = cv.findHomography(point_inp, dst) # src, dst
M = cv.getPerspectiveTransform(point_inp,dst)
#dst = cv.warpPerspective(inp, h, (W,H)) #wraped image

cv.waitKey(0)
show_wait_destroy("homograpghy", dst)
dst = cv.warpPerspective(inp, M, (W,H)) #wraped image
#show_wait_destroy("pers", dst)

#%% detect notes
if len(dst.shape) != 2:
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
else:
    gray = dst

inp_sheet2=np.copy(dst)
gray = cv.bitwise_not(gray)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
horizontal = np.copy(bw)
lines = _lines(horizontal,"h-lines")
sm = remove_lines(bw2, inp_sheet2)
#I didn't like the result of removed lines so lets try cutting the image

sm = remove_lines(bw2, inp_sheet)

vertical = np.copy(bw)
# Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = rows // 30
# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
# Apply morphology operations
vertical = cv.erode(vertical, verticalStructure)
vertical = cv.dilate(vertical, verticalStructure)
# Show extracted vertical lines
cut1 = cut(inp,x, y, w, h1, "cut")
if len(cut1.shape) != 2:
    gray = cv.cvtColor(cut1, cv.COLOR_BGR2GRAY)
else:
    gray = cut1
gray = cv.bitwise_not(gray)
bw1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
#horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
#find lines
Lines = np.copy(bw1)
L = rows // 30
L_Structure = cv.getStructuringElement(cv.MORPH_RECT, (1, L))
detected_lines = cv.morphologyEx(Lines, cv.MORPH_OPEN, L_Structure, iterations=2)
cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv.drawContours(inp_sheet, [c], -1, (255, 255, 255), 2)
# Repair image
repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
result_L = 255 - cv.morphologyEx(255 - cut1, cv.MORPH_CLOSE, repair_kernel, iterations=1)
show_wait_destroy("all_lines", result_L)
thresh = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)[1]

# find contours and get area
# draw all contours in green
contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
#area_thresh = 0
min_area = 0.95*180*35
max_area = 1.05*180*35
result = np.copy(result_L)
for c in contours:
    area = cv.contourArea(c)
    contours= cv.drawContours(result, [c], -1, (0, 255, 0), 1)
    if area > min_area and area < max_area:
        contours= cv.drawContours(result, [c], -1, (0, 0, 255), 1)


# save result
cv.imwrite("box_found.png", result)

# show images
cv.imshow("GRAY", gray)
cv.imshow("THRESH", thresh)
cv.imshow("RESULT", result)
cv.waitKey(0)


#%%%%





# #%% cut them
cut = cut(sm,x, y, w, h1, "cut")
cut2 =np.copy(cut)
cut4 =np.copy(cut)


if len(cut.shape) != 2:
    gray = cv.cvtColor(cut, cv.COLOR_BGR2GRAY)
else:
    gray = cut
gray = cv.bitwise_not(gray)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
circle_inp = np.copy(bw)
img_blur = cv.medianBlur(gray, 5)



## finding all staffs and notes
import random as rng
rng.seed(12345)
def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None] * len(contours)
    minEllipse = [None] * len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    # Draw contours + rotated rects + ellipses

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i, c in enumerate(contours):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        # contour
        cv.drawContours(drawing, contours, i, color)
        box = cv.boxPoints(minRect[i])
        box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv.drawContours(drawing, [box], 0, color)

    show_wait_destroy("counters", drawing)


src = cut2
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
max_thresh = 255
thresh = 100  # initial threshold


pic = thresh_callback(thresh)
cv.waitKey()
pic = cv.bitwise_not(pic)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
pic_blur = cv.blur(bw, (3, 3))
circles = cv.HoughCircles(pic_blur, cv.HOUGH_GRADIENT, 1, cut2.shape[0]/64, param1=200, param2=10, minRadius=10, maxRadius=20)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv.circle(cut2, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv.circle(cut2, (i[0], i[1]), 2, (0, 0, 255), 3)

show_wait_destroy("circles-notes", cut2)


