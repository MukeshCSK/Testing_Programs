import os
import cv2

# Specify the output directory where cropped images will be saved
output_dir = "C:\\Users\\mukes\\Desktop\\CroppedFaces\\"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# read the input image
img = cv2.imread('C:\\Users\\mukes\\Desktop\\Faces.png')

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# read the haarcascade to detect the faces in an image
face_cascade = cv2.CascadeClassifier("C:\\Users\\mukes\\Desktop\\HaarCascade\\haarcascade_frontalface_alt.xml")

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print('Number of detected faces:', len(faces))

# loop over all detected faces
if len(faces) > 0:
   for i, (x,y,w,h) in enumerate(faces):
      
      # To draw a rectangle around the face
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
      face = img[y:y+h, x:x+w]

      # Save the cropped face in the output directory
      output_path = os.path.join(output_dir, f"face{i}.jpg")
      cv2.imwrite(output_path, face)
      print(f"{output_path} is saved")

# display the image with detected faces
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
