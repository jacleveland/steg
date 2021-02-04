import numpy as np
from PIL import Image

'''
****************************************************************
Jacob Cleveland
1/15/21
ECEN 4920
University of Nebraska, Lincoln
Department of Electrical and Computer Engineering

Test file to change the value of a pixel on an image

This serves as a jumping off point and a good first hurdle 

towards a basic implementation of stegonography. (LSB method)
****************************************************************
'''

filename = 'reef.jpeg'
image = Image.open(filename)   #open image file using pillow
print(type(image))
data = np.asarray(image)       #store image data as numpy array
print(type(data[0][0][0]))
print(data.shape)

writeable = np.copy(data)      #copy a mutable copy of the array

print(data[0][0][0])
print(writeable[0][0][0])
writeable[10][0][0] = 243       #write the value to be changed
print(writeable[0][0][0])      #in the proper location
image2 = Image.fromarray(writeable) #generate new image from doctored array
print(type(image2))
image2.show()                       #show the image


