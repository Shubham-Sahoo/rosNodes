import cv2
import dlib
import numpy as np
import time


PREDICTOR_PATH = "/home/hamhub/Downloads/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    cv2.imshow('Result', image_with_landmarks)
    cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    cv2.waitKey(4)
    #cv2.destroyAllWindows()

def top_eye_left(landmarks):
    top_eye_left_pts = []
    top_eye_left_pts.append(landmarks[37])
    top_eye_left_pts.append(landmarks[38])

    top_eye_left_all_pts = np.squeeze(np.asarray(top_eye_left_pts))
    top_eye_left_mean = np.mean(top_eye_left_pts, axis=0)
    return int(top_eye_left_mean[:,1])

def bottom_eye_left(landmarks):
    bottom_eye_left_pts = []
    bottom_eye_left_pts.append(landmarks[40])
    bottom_eye_left_pts.append(landmarks[41])
    bottom_eye_left_all_pts = np.squeeze(np.asarray(bottom_eye_left_pts))
    bottom_eye_left_mean = np.mean(bottom_eye_left_pts, axis=0)
    return int(bottom_eye_left_mean[:,1])

def left_eye_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_eye_left_center = top_eye_left(landmarks)
    bottom_eye_left_center = bottom_eye_left(landmarks)
    eye_left_distance = abs(top_eye_left_center - bottom_eye_left_center)
    return image_with_landmarks, eye_left_distance

    cv2.imshow('Result', image_with_landmarks)
    cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    cv2.waitKey(4)
    #cv2.destroyAllWindows()


def top_eye_right(landmarks):
    top_eye_right_pts = []
    top_eye_right_pts.append(landmarks[43])
    top_eye_right_pts.append(landmarks[44])
    top_eye_right_all_pts = np.squeeze(np.asarray(top_eye_right_pts))
    top_eye_right_mean = np.mean(top_eye_right_pts, axis=0)
    return int(top_eye_right_mean[:,1])

def bottom_eye_right(landmarks):
    bottom_eye_right_pts = []
    bottom_eye_right_pts.append(landmarks[46])
    bottom_eye_right_pts.append(landmarks[47])
    bottom_eye_right_all_pts = np.squeeze(np.asarray(bottom_eye_right_pts))
    bottom_eye_right_mean = np.mean(bottom_eye_right_pts, axis=0)
    return int(bottom_eye_right_mean[:,1])

def right_eye_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_eye_right_center = top_eye_right(landmarks)
    bottom_eye_right_center = bottom_eye_right(landmarks)
    eye_right_distance = abs(top_eye_right_center - bottom_eye_right_center)
    return image_with_landmarks, eye_right_distance

    cv2.imshow('Result', image_with_landmarks)
    cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    cv2.waitKey(4)
    #cv2.destroyAllWindows()





cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 
total = 1
count=1
score=0
starttime=time.time()

while True:
    ret, frame = cap.read()   
    image_landmarks, lip_distance = mouth_open(frame)
    image_landmarks, left_eye_distance = left_eye_open(frame)
    image_landmarks, right_eye_distance = right_eye_open(frame)
    
    prev_yawn_status = yawn_status  
    
    print ("lip distance: ", lip_distance)
    print ("left_eye distance: ", left_eye_distance)
    print ("right_eye distance: ", right_eye_distance)
    if (left_eye_distance and right_eye_distance) < 6:
        count = count + 1
        tick=time.time()-starttime
        if count/tick > 0:
            score += 10
            print ("not alert",tick)  
            
    
        if tick>50:
            starttime=time.time()
            count=0
    

    if lip_distance > 20:
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        output_text = " Yawn Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
        score += 2000


    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(33) == ord('a'):
        break
        
cap.release()
cv2.destroyAllWindows()
