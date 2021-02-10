import numpy as np
from PIL import Image

'''
****************************************************************
Jacob Cleveland
1/19/21
ECEN 4920
University of Nebraska, Lincoln
Department of Electrical and Computer Engineering

Least Significant Bit steganography implementation module.
****************************************************************
'''

'''
****************************************************************
Changes the pixels at (x,y) to the value (R,G,B)
****************************************************************
'''
def changePixel(x, y, R, G, B):
    if((0 <= x < xmax) and (0 <= y < ymax)):
        writeable[y][x][0] = R       #write the red value in pixel (x,y)
        writeable[y][x][1] = G       #green value in pixel (x,y)
        writeable[y][x][2] = B       #blue value in pixel (x,y)
    else:
        print("Coordinates out of bounds! ({},{})".format(x,y))

'''
*********************************************************
Encode a string message in the supplied image

TODO: write so that a selected channel (R,G,B)
is encoded instead of all 3
*********************************************************
'''
def encode(message,channel):
    byte = bytes(message, 'ASCII') #save message as array of ascii bytes
    xval = 0
    yval = 0
    for values in byte:
        counter = 8
        while(counter != 0): #can ignore initial 0 since ascii characters
            onesplace = values % 2
            if(onesplace == 0):
                (R,G,B) = writeable[yval][xval]
                if((R%2 == 1) & (channel == 0)):
                    if(R!=0):
                        R -= 1
                    else:
                        R += 1
                if((G%2 == 1) & (channel == 1)):
                   if(G!=0):
                       G -= 1
                   else:
                       G -= 1
                if((B%2 == 1) & (channel == 2)):
                   if(B!=0):
                       B -= 1
                   else:
                       B -= 1
                changePixel(xval, yval, R,G,B)
            else:
                (R,G,B) = writeable[yval][xval]
                if((R%2 == 0) & (channel == 0)):
                    if(R!=0):
                        R -= 1
                    else:
                        R += 1
                if((G%2 == 0) & (channel == 1)):
                   if(G!=0):
                       G -= 1
                   else:
                       G -= 1
                if((B%2 == 0) & (channel == 2)):
                   if(B!=0):
                       B -= 1
                   else:
                       B -= 1
                changePixel(xval, yval, R,G,B)
            xval+=1
            if(xval >= xmax):
                xval = 0
                yval += 1
            values = int(values/2)
            counter-=1
    for z in range(0,8):
        (R,G,B) = writeable[yval][xval]
        if(R%2 == 1):
            if(R!=0):
                R -= 1
            else:
                R += 1
        if(G%2 == 1):
            if(G!=0):
                G -= 1
            else:
                G -= 1
        if(B%2 == 1):
            if(B!=0):
                B -= 1
            else:
                B -= 1
        changePixel(xval,yval,R,G,B)
        xval +=1
        if(xval >= xmax):
            xval = 0
            yval += 1

'''
**************************************************************
extracts an array of binary digits from a stego image

TODO: rewrite so information is extracted from the correct
channel among (R,G,B)
**************************************************************
'''
def extract(stegoImage,channel):
    result = []
    zero = [0,0,0,0,0,0,0,0]
    y = 0
    x = 0
    buff = []
    while(buff != zero):
       buff = []
       counter = 8
       while(counter != 0):
           digit = int(writeable[y][x][channel]%2)
           x+=1
           if(x==xmax):
               x=0
               y+=1
           buff.append(digit)
           result.append(digit)
           counter-=1
    return result

'''
****************************************************************
Decode a string message from a supplied array of pixel values
****************************************************************
'''
def decode(cover):
    val = []
    count = 0
    while((8*(count+1)) <= len(cover)):
        val.append(0)
        for z in range(count * 8,(count+1)*8):
            val[count] += pow(2,int(z%8))*cover[z]
        count += 1
    result = ""
    for vals in val:
        if(vals != 0):
            result += chr(vals)
    return result

'''
*******
Main
*******
'''

filename = 'reef.jpeg'         #cover file jpeg to be used
channel = 0                    #R=0, G=1, B=2; channel used for embedding
image = Image.open(filename)   #open image file using pillow
data = np.asarray(image)       #store image data as numpy array
writeable = np.copy(data)      #copy a mutable copy of the array
xmax = writeable.shape[1]      #x array bound
ymax = writeable.shape[0]      #y array bound

g = open('message.txt','r')
message = g.readline()
g.close()
encode(message,channel)
extracted = extract(writeable,channel)
decoded = decode(extracted)
print(decoded)

image2 = Image.fromarray(writeable) #generate new image from doctored array
image2.show()                       #show the image

