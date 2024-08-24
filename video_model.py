import cv2
import os

# Load the Haar cascade for license plate detection
license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Path to the video file
video_path = 'Video_data.mp4'

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
else:
    # Capture video from file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print('Error: Unable to open the video file.')
    else:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("End of video or error reading the frame.")
                break
            # Resize the frame to make it smaller
            scale_percent = 50  # Percentage of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))
            
            # Convert the frame to grayscale for detection
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect license plates in the frame
            license_plates = license_plate_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))
            for (x, y, width, height) in license_plates:
                # Draw a rectangle around the detected license plate
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                # Blur the license plate area
                frame[y:y + height, x:x + width] = cv2.blur(frame[y:y + height, x:x + width], ksize=(10, 10))
                # Add a label above the rectangle
                cv2.putText(frame, text='License Plate', org=(x - 3, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=1, fontScale=0.6)
            # Display the processed video frame
            cv2.imshow('Video', frame)
            # Exit the loop when 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        # Release the video capture object and close all OpenCV windows
        video_capture.release()
        cv2.destroyAllWindows()
        

