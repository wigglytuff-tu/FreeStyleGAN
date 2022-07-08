import numpy as np

# eye and mouth positions in 3D canonical space
eyes3D =  np.array([[-1, 0, 0, 1], [1, 0, 0, 1]])
mouthZ = 0.25 # this is 1/8 the interocular distance

# x_- and x_+
referenceAlignmentFeats = np.array([[-1., 0.], [ 1., 0.]])

# x_- and x_+ in absolute pixel coordinates
def getReferencePixelAlignmentFeats(res=1024):
    return (0.5 * res * (referenceAlignmentFeats + 1.)).astype(np.int32)