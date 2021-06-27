import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") ## Taking the instructions from opencv

skip = 0
face_data = []
dataset_path = "./face_dataset/" ## Path to save the face data

file_name = input("Enter the name of person : ") ## Console message...

while True:
    ret, frame = cap.read() ## Open the camera. 

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ## Frame gray scale.

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) ## frame_name, scaling factor, k(no. of neighbors)
    if len(faces) == 0:
        continue

    k = 1

    faces = sorted(faces, key = lambda x : x[2]*x[3], reverse = True)

    skip += 1

    for face in faces[:1]:
        x,y,w,h = face

        offset = 5
        face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection = cv2.resize(face_offset,(100,100))

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        cv2.imshow(str(k), face_selection)
        k += 1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("faces",frame)

    key_pressed = cv2.waitKey(1)  & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + file_name, face_data)
print("Dataset Saved At : {}".format(dataset_path + file_name + '.npy'))

cap.release()
cv2.destroyAllWindows()