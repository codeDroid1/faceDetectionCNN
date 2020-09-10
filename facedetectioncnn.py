# -*- coding: utf-8 -*-
'''
!pip install mtcnn
!pip install keras_vggface
'''

from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle

image = plt.imread('../media/face-001.jpg')

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
        plt.savefig('../media/face_detected.jpg')
    plt.show()

"""## Highlight faces from an Image"""

highlight_faces('../media/face-001.jpg', faces)

