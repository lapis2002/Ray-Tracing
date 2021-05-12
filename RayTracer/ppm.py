import numpy as np
from PIL import Image
'''
Output in P6 format, a binary file containing:
P6
ncolumns nrows
Max colour value
colours in binary format thus unreadable
@param: (string) fname
@param: (numpy array) im
'''
def savePpmP6 (fname, im):
    im = (im*255).astype(np.uint8)

    imgPpm = Image.fromarray(im)
    imgPpm.save(fp=fname)

def savePpmP3 (fname, im):
    maxVal = 255
    im = im*255
    height, width, nc = im.shape
    assert nc == 3
     
    f = open(fname, 'w')
     
    f.write('P3\n')
    f.write(str(width)+' '+str(height)+'\n')
    f.write(str(maxVal)+'\n')
     
    # interleave image channels before streaming    
    c1 = np.reshape(im[:, :, 0], (width*height, 1))
    c2 = np.reshape(im[:, :, 1], (width*height, 1))
    c3 = np.reshape(im[:, :, 2], (width*height, 1))
     
    im1 = np.hstack([c1, c2, c3])
    im2 = im1.reshape(width*height*3)
    
    f.write('\n'.join(im2.astype('int').astype('str')))
    f.write('\n')
 
    f.close()

