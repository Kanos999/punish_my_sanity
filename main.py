import numpy as np
import cv2 as cv

cv.namedWindow('Controls')

cap = cv.VideoCapture(0)

def nothing(x):
    pass


cv.createTrackbar('H_lower', 'Controls', 47, 179, nothing)
cv.createTrackbar('H_upper', 'Controls', 74, 179, nothing)

cv.createTrackbar('S_lower', 'Controls', 87, 255, nothing)
cv.createTrackbar('S_upper', 'Controls', 255, 255, nothing)

cv.createTrackbar('V_lower', 'Controls', 77, 255, nothing)
cv.createTrackbar('V_upper', 'Controls', 255, 255, nothing)

timeTillStart = 30
drinkingCounter = 150
drinkingMode = False

while cap.isOpened():


    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    lower_green = np.array([47,87,77])
    upper_green = np.array([74,255,255])

    # get current positions of trackbars
    lower_green[0] = cv.getTrackbarPos('H_lower','Controls')
    lower_green[1] = cv.getTrackbarPos('S_lower','Controls')
    lower_green[2] = cv.getTrackbarPos('V_lower','Controls')

    upper_green[0] = cv.getTrackbarPos('H_upper','Controls')
    upper_green[1] = cv.getTrackbarPos('S_upper','Controls')
    upper_green[2] = cv.getTrackbarPos('V_upper','Controls')


    blur = cv.medianBlur(frame,15,1)
    blur = cv.GaussianBlur(blur,(15,15),1)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_green, upper_green)
    #mask = cv.GaussianBlur(mask,(35,35),0)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask = mask)

    # Finding the largest cluster of orange pixels
    # using findContours func to find the none-zero pieces
    _, contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


    sumX = 0
    sumY = 0
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        #if w*h < 100: continue
        sumX += x + w/2
        sumY += y + h/2

    if len(contours) != 0:
        print(sumX)
        avgX = int(sumX / len(contours))
        avgY = int(sumY / len(contours))

        radius = 20
        res = cv.circle(res, (avgX,avgY), radius, (0,255,0), 2)

    cv.imshow('result', res)


    # if largest != (0,0,0,0):
    #     x,y,w,h = largest
    #     cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    #     timeTillStart += -1
    #     if timeTillStart <= 0: drinkingMode = True
    # else:
    #     timeTillStart = 30

    
    frame = cv.rectangle(frame, (0,0), (240,60), (20,20,20), -1)

    
    if drinkingMode:
        cv.putText(
            frame, #numpy array on which text is written
            f"TIME REMAINING: {int(drinkingCounter/3)/10}", #text
            (20,40), #position at which writing has to start
            cv.FONT_HERSHEY_SIMPLEX, #font family
            0.6, #font size
            (0, 0, 255), #font color
            2) #font stroke
        drinkingCounter += -1
        if drinkingCounter <= 0: 
            drinkingMode = False
            drinkingCounter = 150
    else:
        cv.putText(
            frame, #numpy array on which text is written
            f"Time till start: {timeTillStart}", #text
            (20,40), #position at which writing has to start
            cv.FONT_HERSHEY_SIMPLEX, #font family
            0.6, #font size
            (0, 255, 0), #font color
            2) #font stroke

    # show the shit
    #out.write(frame)
    cv.imshow('frame', frame)
    
    k = cv.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('r'):
        bounces = 0
    
    
# Release everything if job is finished
cap.release()
#out.release()
cv.destroyAllWindows()

