import cv2


def load_plate_cascade():
    """
    Load the Haar Cascade for license plate detection.
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    return cv2.CascadeClassifier(cascade_path)

def read_image(image_path):
    """
    Read the image from the specified path.
    """
    return cv2.imread(image_path)

def convert_to_grayscale(image):
    """
    Convert the image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_plates(gray_image, plate_cascade):
    """
    Detect license plates in the grayscale image.
    """
    return plate_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(25, 25)
    )

def process_detected_plates(image, plates):
    """
    Process and annotate detected plates in the image.
    """
    for (x, y, w, h) in plates:
        # Draw a rectangle around the plate
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        # Blur the detected plate region
        image[y:y+h, x:x+w] = cv2.blur(image[y:y+h, x:x+w], ksize=(10, 10))
        # Add text label above the rectangle
        cv2.putText(
            image, 
            'License Plate', 
            org=(x, y-10), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX, 
            fontScale=0.6, 
            color=(0, 0, 255), 
            thickness=1
        )

def display_image(image, window_name='Automatic Liscence Plate Recognition System'):
    """
    Display the processed image in a window.
    """
    cv2.imshow(window_name, image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def main(image_path):
    """
    Main function to load, process, and display the image with detected plates.
    """
    plate_cascade = load_plate_cascade()
    image = read_image(image_path)
    gray_image = convert_to_grayscale(image)
    plates = detect_plates(gray_image, plate_cascade)
    process_detected_plates(image, plates)
    display_image(image)

if __name__ == '__main__':
    # Replace 'img.jpg' with the path to your image file
    main('Image_data.jpg')
