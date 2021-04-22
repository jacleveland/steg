# steg
Steganography projects completed as part of my independent study with Dr. Dongming Peng. 
Implementations include Least Significant Bit (LSB), Quantization Index Modulation (QIM), and an Artificial Neural Network (ANN) approach.

The LSB file is lsb.py which requires reef.png as well as message.txt

The QIM file is qim3.py which also requires reef.png as well as message.txt

The ANN file is nnsteg2.py which doesn't require any additional files, 
but the dependencies include pytorch among other things.
Example input is alt.png and example output is stegimage.png which
can be seen to be essentially identical despite stegimage containing
the message "HELLOWORLD"
Note that the output of this is an array of floating point values corresponding
to the characters in the message.
The conversion is as follows:
A is any value in [-150,-140)
B is any value in [-140,-130)
        .
        .
        .
Y is any value in [90,100)
Z is any value in [100,110]
