import cv2 as cv

def main():
    '''
    img = cv.imread('img/fotolinkedin.jpeg')
    cv.imshow('Yo Fachero',img)
    cv.waitKey(0)
    '''

    #Traigo un modelo de cara
    face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    cap = cv.VideoCapture(0)

    while True:
        #Captura Frame p/ Frame
        ret, frame = cap.read()

        #Seteo la cam en blanco y negro y busco parentezco entre cara
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        #Traigo las coordenadas de las caras
        for (x,y,w,h) in faces:
            print(f"Cara en coords: {x,y,w,h}")
            roi_gray = gray[y:y+h, x:x+w]
            img_item = 'my-image.png'
            cv.imwrite(img_item, roi_gray) #Solo guardo mi cara

            color = (0,255,0)
            stroke = 2
            end_coord_x = x+w
            end_coord_y = y+h
            cv.rectangle(frame, (x,y), (end_coord_x,end_coord_y), color, stroke)

        #Muestro la camara
        cv.imshow('Camara',frame)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyWindow()

    return True

main()