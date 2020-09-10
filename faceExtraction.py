from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle

image = plt.imread('face-001.jpg')

detector = MTCNN()

faces = detector.detect_faces(image)
for face in faces:
    print(face)

def highlight_faces(image_path, faces):
  # display image
    image = plt.imread(image_path)
    plt.imshow(image)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                          fill=False, color='red')
        ax.add_patch(face_border)
        plt.savefig('face_detected.jpg')
    plt.show()

"""## Highlight faces from an Image"""

highlight_faces('face-001.jpg', faces)

"""## Extract Face from image"""

from numpy import asarray
from PIL import Image

def extract_face_from_image(image_path, required_size=(224, 224)):
  # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images

extracted_face = extract_face_from_image('face-001.jpg')

# Display the first face from the extracted faces]
plt.subplot(121),plt.imshow(extracted_face[0]),plt.title("first face")
plt.subplot(122),plt.imshow(extracted_face[1]),plt.title("second face")
plt.savefig("extracted face.jpg")
plt.show()

