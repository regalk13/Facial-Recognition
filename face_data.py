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

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) # Detect multi faces in the image
    """detectMultisacle params: frame_name, scaling factor, k(no. of neighbors),
    returns: x, w, z, h coordinates of the faces on a list."""


    if len(faces) == 0:
        continue

    k = 1

    faces = sorted(faces, key = lambda x : x[2]*x[3], reverse = True) ## faces = [x, w, z, h], area = w*h = faces[2] * faces[3], area of the face.

    skip += 1 ## Increment every time, takes the face.

    for face in faces[:1]:
        x,y,w,h = face

        offset = 5
        face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset] ## Taking offset of the coordinates like a padding.
        face_selection = cv2.resize(face_offset,(100,100)) ## Resizing the face to 100x100.

        if skip % 10 == 0: ## Every 10 entries, recording 10 faces to the array.
            face_data.append(face_selection)
            print(len(face_data))

        cv2.imshow(str(k), face_selection)
        k += 1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) ## Rectangle coordinates, color RGB. So this show a green box containing face.
    
    cv2.imshow("faces",frame)

    key_pressed = cv2.waitKey(1)  & 0xFF ## If press 'q' the camera is off.
    if key_pressed == ord('q'):
        break

face_data = np.array(face_data) ## Saving the faces data on a np array.
face_data = face_data.reshape((face_data.shape[0], -1)) ## Reshaping the np array, (read  docs for more information).
print(face_data.shape)

np.save(dataset_path + file_name, face_data) ## Saving the saved data like .npy on the path.
print("Dataset Saved At : {}".format(dataset_path + file_name + '.npy'))

cap.release()
cv2.destroyAllWindows() ## Close the camera.