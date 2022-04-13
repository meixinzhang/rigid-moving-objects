import numpy as np
from skimage.feature import corner_harris, corner_peaks, BRIEF, match_descriptors


def get_matched_keyppoints(imLgray, imRgray):
    """
    Takes a pair of gray scale images and returns matching features from them.
    (based on harrison corners and peaks)
    """
    keypointsL = corner_peaks(corner_harris(imLgray), threshold_rel=0.001, min_distance=15)
    keypointsR = corner_peaks(corner_harris(imRgray), threshold_rel=0.001, min_distance=15)
    print ('the number of features in images 1 and 2 are {:5d} and {:5d}'.format(keypointsL.shape[0], keypointsR.shape[0]))

    extractor = BRIEF()
    extractor.extract(imLgray, keypointsL)
    keypointsL = keypointsL[extractor.mask]         
    descriptorsL = extractor.descriptors
    extractor.extract(imRgray, keypointsR)
    keypointsR = keypointsR[extractor.mask]
    descriptorsR = extractor.descriptors

    matchesLR = match_descriptors(descriptorsL, descriptorsR, cross_check=True)
    print ('the number of matches is {:2d}'.format(matchesLR.shape[0]))
    
    return keypointsL, keypointsR, matchesLR

def matchedpoints_to_pairpoints(keypointsL, keypointsR, matchesLR, inv=True):
    """
    Take key points in the left image and right image as well as an array for matched features of shape (n, 2),
    returns two arrays where ptsL[i] and ptsR[i] are a pair of matched features in left and right image

    inv is a optional parameter to indicate whether or not we want to swap the two columns of the output points
    """
    ptsL = []
    ptsR = [] 
    for i in matchesLR:
        ptsL.append(keypointsL[i[0]])
        ptsR.append(keypointsR[i[1]])

    ptsL = np.array(ptsL)
    ptsR = np.array(ptsR)
    if inv:
        # swapping columns using advanced indexing https://docs.scipy.org/doc/numpy/reference/arra
        # This changes point coordinates from (y,x) in ptsL1/ptsR1 to (x,y) in ptsL/ptsR
        # Since some feature extractor will output coordinate (y,x)
        return ptsL[:,[1, 0]], ptsR[:,[1, 0]]
    
    return ptsL, ptsR
