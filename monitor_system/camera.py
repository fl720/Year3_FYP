import cv2 

def capture_frame():
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open the camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        print("Failed.")
        return None
    
if __name__ == '__main__':
    filename = 'saved_frame.jpg'
    img = capture_frame() 
    cv2.imwrite(filename, img)