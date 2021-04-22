import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import math
from skimage.io import imread
import matplotlib.pylab as plt

'''
****************************************************************
Jacob Cleveland
3/23/21
ECEN 4920
University of Nebraska, Lincoln
Peter Kiewit Institute
Department of Electrical and Computer Engineering

Quantization Index Modulation steganography implementation module.
****************************************************************
'''

'''
****************
Implement 2D DCT
****************
'''
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

'''
*****************
Implement 2D IDCT
*****************
'''
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

'''
******************************************
Dual purpose function:
Split YCbCr array into each channel,
and expand into blockable (div. by 8) size
by extending the right and bottom.
******************************************
'''
def ycbcr(a,shape,channel):
    b = []
    for y in range(0,shape[0]):
        buf = []
        for x in range(0,shape[1]):
            buf.append(a[y][x][channel])
        for z in range(shape[1],shape[1]+8-(shape[1]%8)):
            buf.append(a[y][x][channel])
        b.append(buf)
    for z in range(shape[0],shape[0]+8-(shape[0] % 8)):
        b.append(buf)
    return np.copy(b)

def expand(a):
    b = []
    shape = a.shape
    for u in range(0,shape[0]):
        buf = []
        for v in range(0,shape[1]):
            buf.append(a[u][v])
        for w in range(shape[1],shape[1]+8-(shape[1]%8)):
            buf.append(a[u][v])
        b.append(buf)
    for z in range(shape[0],shape[0]+8-(shape[0]%8)):
        b.append(buf)
    return np.copy(b)

def cut(a,shape):
    b = []
    for u in range(0,shape[0]):
        inside = []
        for v in range(0,shape[1]):
            inside.append(a[u][v])
        b.append(inside)
    return np.copy(b)

def recombine(Y,Cb,Cr,shape):
    b = []
    for row in range(0,shape[0]):
        inside = []
        for col in range(0,shape[1]):
            inside.append([int(Y[row][col]),int(Cb[row][col]),int(Cr[row][col])])
        b.append(inside)
    b = np.copy(b)
    return b

def quantize(a,qtable):
    b = []
    shape = a.shape
    for u in range(0,shape[0]):
        buf = []
        for v in range(0,shape[1]):
            buf.append(int(a[u][v]/qtable[u%8][v%8]))
        b.append(buf)
    return np.copy(b)

def modulate(a,nonemb,message):
    b = np.copy(a).astype('float64')
    shape = b.shape
    mcount = 0
    mlen = len(message)
    stop = 8
    breakout = False
    lastu = 0
    lastv = 0
    for blockh in range(0,int(shape[0]/8)):
        if(mcount < mlen):
            for blockv in range(0,int(shape[1]/8)):
                if(mcount < mlen):
                    contend = [100]
                    for u in range(nonemb,8):
                        for v in range(nonemb,8):
                            if((u,v)!=(0,0)):
                                val = abs(b[u+8*blockh][v+8*blockv])
                                if(val != 0):
                                    contend.append(val)
                    step = 50
                    for u in range(0,nonemb):
                        for v in range(0,nonemb):
                            if(((u,v)!=(0,0))& (mcount < mlen)):
                                oldc = b[u+8*blockh][v+8*blockv]
                                newc = step*math.floor(oldc / step)+step/2*message[mcount]
                                b[u+8*blockh][v+8*blockv] = newc
                                mcount += 1
                                lastu = u
                                lastv = v
                            elif((u,v)!=(0,0)):
                                oldc = b[u+8*blockh][v+8*blockv]
                                newc = step*math.floor(oldc / step)
                                b[u+8*blockh][v+8*blockv] = newc
    return np.copy(b)

def demodulate(a,nonemb):
    steps = []
    shape = a.shape
    message = []
    mcount = 0
    stop = 0
    for blockh in range(0,int(shape[0]/8)):
        for blockv in range(0,int(shape[1]/8)):
            contend = [100]
            for u in range(nonemb,8):
                for v in range(nonemb,8):
                    if((u,v)!=(0,0)):
                        val = abs(a[u+8*blockh][v+8*blockv])
                        if(val != 0):
                            contend.append(val)
            step = 50
            if(stop != 8):
                steps.append(step)
            if(step!=0):
                for u in range(0,nonemb):
                    for v in range(0,nonemb):
                        if((u,v)!=(0,0)):
                            if(step != 0):
                                if(stop != 8):
                                    c = a[u+8*blockh][v+8*blockv]
                                    zero = step * math.floor(c/step)
                                    one = zero + step/2
                                    if(abs(c-zero) < abs(c-one)):
                                        message.append(0)
                                        stop += 1
                                    else:
                                        message.append(1)
                                        stop = 0
                                else:
                                    return message[0:len(message)-7]
                                    break
    return message[0:len(message)-7]

def dequantize(a,qtable):
    b = []
    shape = a.shape
    for u in range(0,shape[0]):
        buf = []
        for v in range(0,shape[1]):
            buf.append(int(a[u][v]*qtable[u%8][v%8]))
        b.append(buf)
    return np.copy(b)

def bytesToArray(bytestream):
    b = []
    for char in bytestream:
        counter = 8
        buff = []
        while(counter != 0):
            buff.append(char % 2)
            char = int(char/2)
            counter -= 1
        for i in range(0,8):
            b.append(buff[7-i])
    return b

'''
*************************************************
Decode takes a big endian array of bits
and converts each byte into its ASCII equivalent.
*************************************************
'''
def decode(bitArray):
    val = []
    length = len(bitArray)
    counter = 0
    result = ""
    while(8*(counter+1) < length):
        val.append(0)
        for i in range(8*counter, 8*(counter+1)):
            val[counter] += pow(2,7-int(i%8))*bitArray[i]
        counter += 1
    for vals in val:
        if(vals != 0):
            result += chr(vals)
    return result

'''
****
main
****
'''
filename = "stegreef1.jpeg"
im = imread(filename)
imshape = im.shape
luma = ycbcr(im,im.shape,0)
cb = ycbcr(im,im.shape,1)
cr = ycbcr(im,im.shape,2)
luma = cut(luma,imshape)
cb = cut(cb,imshape)
cr = cut(cr,imshape)
lumaF = dct2(luma)
cbF = dct2(cb)
crF = dct2(cr)
luma = expand(lumaF)
qtable = [[16,11,10,16,24,40,51,61],
          [12,12,14,19,26,58,60,55],
          [14,13,16,24,40,57,69,56],
          [14,17,22,29,51,87,80,62],
          [18,22,37,56,68,109,103,77],
          [24,35,55,64,81,104,113,92],
          [49,64,78,87,103,121,120,101],
          [72,92,95,98,112,100,103,99]]

luma = quantize(lumaF,qtable)
nonEmbeddingArea = 6
g = open('message.txt','r')
message = g.readline()
g.close()
mbytes = bytes(message,'ASCII')
marray = bytesToArray(mbytes)
luma = modulate(luma,nonEmbeddingArea,marray)
demessage = demodulate(luma,nonEmbeddingArea)
txtmessage = decode(demessage)
print(txtmessage)
luma = dequantize(luma,qtable)
luma = cut(luma,imshape)
lumaF = idct2(lumaF)
cb = idct2(cbF)
cr = idct2(crF)
fshape = lumaF.shape
for u in range(0,fshape[0]):
    for v in range(0,fshape[1]):
        lumaF[u][v] = round(lumaF[u][v])
lumaF = lumaF.astype('uint8')
colour = recombine(lumaF,cb,cr,imshape)
colour = colour.astype('uint8')
stegimage = Image.fromarray(colour)
stegimage.save('qimreef.jpeg')
filename = 'qimreef.jpeg'
image = Image.open(filename)   #open image file using pillow
data = np.asarray(image)       #store image data as numpy array
qim = np.copy(data)      #copy a mutable copy of the array
stegimage = Image.fromarray(qim)
stegimage.show()
qimshape = qim.shape
qluma = ycbcr(qim,qimshape,0)
cb = ycbcr(qim,qimshape,1)
cr = ycbcr(qim,qimshape,2)
qimshape = lumaF.shape
qluma = cut(lumaF,qimshape)
qcb = cut(cb,qimshape)
qcr = cut(cr,qimshape)
qlumaF = dct2(qluma)
qcbF = dct2(qcb)
qcrF = dct2(qcr)
qluma = expand(qlumaF)
qtable = [[16,11,10,16,24,40,51,61],
          [12,12,14,19,26,58,60,55],
          [14,13,16,24,40,57,69,56],
          [14,17,22,29,51,87,80,62],
          [18,22,37,56,68,109,103,77],
          [24,35,55,64,81,104,113,92],
          [49,64,78,87,103,121,120,101],
          [72,92,95,98,112,100,103,99]]

qluma = quantize(qlumaF,qtable)
demessage = demodulate(qluma,nonEmbeddingArea)
txtmessage = decode(demessage)
