import numpy as np
import cv2
from network import loadModel
import math

class Number:

    def __init__(self,x,y,sabran):
        self.x = x
        self.y = y
        self.sabran = sabran


def provera(x,y):
    for broj in brojevi:
        distance = math.sqrt(math.pow(x-broj.x,2) + math.pow(y-broj.y,2))
        if distance < 20:
            return broj
    return None

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def select_roi(image_orig, image_bin,linije):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    for x1, y1, x2, y2 in linije[0]:
        doleX = x1
        doleY = y1
        goreX = x2
        goreY = y2

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 15 and h < 50 and h > 13 and w > 3:
            broj = provera(x,y)
            if broj is None:
                broj = Number(x, y, False)
                brojevi.append(broj)
            else:
                broj.x = x
                broj.y = y
            if x >= doleX and x<= goreX and y <= doleY and y >= goreY:
                if y - doleY >= ((goreY - doleY) / (goreX - doleX)) * (x - doleX):
                    if broj.sabran is False:
                        broj.sabran = True
                        region = image_bin[y:y + h , x:x + w]
                        regions_array.append([resize_region(region), (x, y, w, h)])
                        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_orig, regions_array

def HoughLinesTransf(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    imgts = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]
    minLineLength = 100
    maxLineGap = 100
    lines = cv2.HoughLinesP(imgts, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    return img,edges,lines

brojevi =[]
model = loadModel()
cap = cv2.VideoCapture("dataset/video-9.avi")
ret, frame = cap.read()
b, g, r = cv2.split(frame)
bb = cv2.GaussianBlur(b, (7, 7), 0)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(bb, kernel, iterations=1)


slika,ivica,linije = HoughLinesTransf(erosion)
cv2.imwrite('houghlines5.jpg', erosion)
suma = 0
while (True):
    ret, frame = cap.read()
    if ret is not True :
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    imgb = image_bin(gray)

    slika, region = select_roi(frame, imgb,linije)
    for r in region:
        broj = r[0].reshape(1,1,28,28)
        maxActivation = 0
        number = 0
        index = 0
        nesto = model.predict(broj)
        for activation in nesto[0]:
            if activation > maxActivation :
                maxActivation = activation
                number = index
            index=index+1
        suma = suma + number


    #cv2.line(frame, (doleX, doleY), (goreX, goreY), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    print('Suma je ' + suma.__str__())

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()