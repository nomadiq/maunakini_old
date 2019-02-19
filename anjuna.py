import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import math
from pathlib import Path

# ------------------------------------- #
#              Functions                #
# ------------------------------------- #


def make_complex(data):

    return data[..., ::2] + data[..., 1::2] * 1.j


def next_fourier_number(num):

    return math.ceil(math.log(num, 2))


def remove_bruker_filter(data, grpdly):

    s = float(data.shape[-1])
    data = np.fft.fft(np.fft.ifftshift(data, -1), axis=-1).astype(data.dtype) / s
    data = data * np.exp(2.j * np.pi * grpdly * np.arange(s) / s).astype(data.dtype)
    data = np.fft.fftshift(np.fft.ifft(data, axis=-1).astype(data.dtype), -1) * s
    skip = int(np.floor(grpdly + 2.))    
    add = int(max(skip - 6, 0))           
    data[..., :add] = data[..., :add] + data[..., :-(add + 1):-1]
    data = data[..., :-skip]

    return data


def plot_2d_spectrum(spectrum, noise, sign=None):
    
    cmap = matplotlib.cm.Reds_r   # contour map (colors to use for contours)
    contour_start = noise     # contour level start value
    contour_num = 10        # number of contour levels
    contour_factor = 1.40      # scaling factor between contour levels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # plot the contours
    ax.contour(spectrum,
               cl,
               cmap=cmap,
               extent=(0, spectrum.shape[1] - 1, 0, spectrum.shape[0] - 1),
               linewidths=1)

    if sign=='PosNeg':
        cl = -1*contour_start * contour_factor ** np.arange(contour_num)
        cmap = matplotlib.cm.Greens
        ax.contour(spectrum, cl[::-1],
                   cmap=cmap,
                   extent=(0, spectrum.shape[1] - 1, 0, spectrum.shape[0] - 1),
                   linewidths=1,
                   )
    
    plt.show()


def plot_2d_nuslist(nuslist):

    x = []
    y = []

    for samp in nuslist:
        x.append(samp[0])
        y.append(samp[1])

    plt.scatter(x, y)


def dd2g(dspfvs, decim):

    dspdic = {
        10: {
            2: 44.75,
            3: 33.5,
            4: 66.625,
            6: 59.083333333333333,
            8: 68.5625,
            12: 60.375,
            16: 69.53125,
            24: 61.020833333333333,
            32: 70.015625,
            48: 61.34375,
            64: 70.2578125,
            96: 61.505208333333333,
            128: 70.37890625,
            192: 61.5859375,
            256: 70.439453125,
            384: 61.626302083333333,
            512: 70.4697265625,
            768: 61.646484375,
            1024: 70.48486328125,
            1536: 61.656575520833333,
            2048: 70.492431640625,
            },
        11: {
            2: 46.,
            3: 36.5,
            4: 48.,
            6: 50.166666666666667,
            8: 53.25,
            12: 69.5,
            16: 72.25,
            24: 70.166666666666667,
            32: 72.75,
            48: 70.5,
            64: 73.,
            96: 70.666666666666667,
            128: 72.5,
            192: 71.333333333333333,
            256: 72.25,
            384: 71.666666666666667,
            512: 72.125,
            768: 71.833333333333333,
            1024: 72.0625,
            1536: 71.916666666666667,
            2048: 72.03125
            },
        12: {
            2: 46.,
            3: 36.5,
            4: 48.,
            6: 50.166666666666667,
            8: 53.25,
            12: 69.5,
            16: 71.625,
            24: 70.166666666666667,
            32: 72.125,
            48: 70.5,
            64: 72.375,
            96: 70.666666666666667,
            128: 72.5,
            192: 71.333333333333333,
            256: 72.25,
            384: 71.666666666666667,
            512: 72.125,
            768: 71.833333333333333,
            1024: 72.0625,
            1536: 71.916666666666667,
            2048: 72.03125
            },
        13: {
            2: 2.75,
            3: 2.8333333333333333,
            4: 2.875,
            6: 2.9166666666666667,
            8: 2.9375,
            12: 2.9583333333333333,
            16: 2.96875,
            24: 2.9791666666666667,
            32: 2.984375,
            48: 2.9895833333333333,
            64: 2.9921875,
            96: 2.9947916666666667
            }
        }
    return dspdic[dspfvs][decim]

# ------------------------------------- #
#               Classes                 #
# ------------------------------------- #

# define class for NUS data - this includes the NUS schedule and assumes Bruker ser file
# the object contains the entire serial file data and the NUS schedule
# points_in_fid is the total (R+I) points in the direct FIDs


class NUSData:

    def __init__(self, data_dir='.', ser_file='ser', nuslist='nuslist',
                 points=None, decim=None, dspfvs=None, grpdly=None):
        
        p0 = dec = dsp = grp = 0
        my_file = Path(data_dir+"/acqus")
        if my_file.is_file():
            acqus_file = open(data_dir+"/acqus", "r")
            for line in acqus_file:
                if "##$TD=" in line: 
                    (name, value) = line.split()
                    p0 = int(int(value)/2)
                if "##$DECIM=" in line:
                    (name, value) = line.split()
                    dec = int(value)
                if "##$DSPFVS=" in line:
                    (name, value) = line.split()
                    dsp = int(value)
                if "##$GRPDLY=" in line:
                    (name, value) = line.split()
                    grp = float(value)
                
            acqus_file.close()

        if grp and not grpdly:
            grpdly = grp

        elif decim and dspfvs:
            grpdly = dd2g(dspfvs, decim)

        elif dec and dsp:
            grpdly = dd2g(dsp, dec)

        if not points:
            points_in_direct_fid = p0*2
        else:
            points_in_direct_fid = points

        print('Number of R+I points: '+str(points_in_direct_fid))
        print('DECIM= '+str(decim)+' DSPFVS= '+str(dspfvs)+' GRPDLY= '+str(grpdly))
        
        # we need ot know how many points are in the direct dimension
        self.pointsInDirectFid = points_in_direct_fid

        # lets open and parse the nuslist file of samples taken
        with open(data_dir+'/'+nuslist, 'r') as nuslist_file:
            lines = nuslist_file.readlines()
            nuslist = []
            for line in lines:
                point = line.split()
                coordinate = []
                for coord in point:
                    coordinate.append(int(coord))
                nuslist.append(coordinate)
        self.nusList = np.array(nuslist)               
        self.nusDimensions = len(nuslist[0]) 
        self.nusPoints = len(nuslist)
        
        # we also want some way to know the order of the samples. This generates indexes
        # that give ordered samples from nuslist based on first column being fast dimension
        # self.ordered_nuslist_index = np.lexsort((self.nuslist[:,0], self.nuslist[:,1]))
        
        # lets load in the actual serial data
        with open(data_dir+'/'+ser_file, 'rb') as serial_file:
            self.nusData = np.frombuffer(serial_file.read(), dtype='<i4')

        # bruker data is four bytes per point so
        # len(nus_data) should equal 4 * 2**self.nus_dimensions * self.nus_points * self.points_in_direct_fid
        if 4 * 2**self.nusDimensions * self.nusPoints * self.pointsInDirectFid == len(self.nusData):
            self.sane = True
        else:
            self.sane = False

        # reshape the data
        self.nusData = np.reshape(self.nusData, (self.pointsInDirectFid, 4, self.nusPoints), order='F')
        self.points = len(remove_bruker_filter(make_complex(np.copy(self.nusData[:, 0, 0])), grpdly))
        print('converted points: ', self.points)
        self.convertedNUSData = np.zeros((self.points, 4, self.nusPoints), dtype='complex128')

        # remove bruker filter
        self.convert_bruker(grpdly)

        self.orderedNUSlistIndex = np.lexsort((self.nusList[:, 0], self.nusList[:, 1]))

    def truncate(self, trunc):

        """
        This function truncates the nusData, nusList and convertedNUSData variables to have only 
        'trunc' number of sampled points
        """

        self.nusList = self.nusList[0:trunc]
        self.nusData = self.nusData[:, :, 0:trunc]
        self.convertedNUSData = self.convertedNUSData[:, :, 0:trunc]
        self.nusPoints = len(self.nusList)
    
    def convert_bruker(self, grpdly):
        # edit the number of points in first dimension after Bruker filter removal
        # self.points = len(remove_bruker_filter(make_complex(np.copy(self.nusData[:, 0, 0])), grpdly))
        # zero fill in a 3D matrix with complex zeros
        # load the data
        for ii in range(self.nusPoints):  #
            for i in range(4):
                fid = remove_bruker_filter(make_complex(np.copy(self.nusData[:, i, ii])), grpdly)
                self.convertedNUSData[0:len(fid), i, ii] = fid

    def order_data(self):
        # we want some way to know the order of the samples. This generates indexes
        # that give ordered samples from nuslist based on first column being fast dimension

        # print(self.nusList[self.ordered_nuslist_index])
        # orderedData = np.zeros( 2**self.nusDimensions * self.nusPoints * self.pointsInDirectFid)

        ordered_data = np.zeros((self.pointsInDirectFid, 4, self.nusPoints), dtype='int64')
        ordered_converted_data = np.zeros((self.points, 4, self.nusPoints), dtype='complex128')

        i = 0
        for point in self.orderedNUSlistIndex:

            ordered_data[:, :, i] = self.nusData[:, :, point]
            ordered_converted_data[:, :, i] = self.convertedNUSData[:, :, point]
            i += 1
        
        # now set the object attributes to the ordered state
        self.nusData = ordered_data
        self.convertedNUSData = ordered_converted_data
        self.nusList = self.nusList[self.orderedNUSlistIndex]
    
    def write_ser(self, file):
        f = open(file, 'wb')
        f.write(self.nusData.astype('<i4').tostring())
        
    def write_nuslist(self, file):
        # f = open(file, 'w')
        # f.write(str(self.nuslist))
        np.savetxt(file, self.nusList, fmt='%i', delimiter='\t')


# define class for Linear bruker data - this includes the number of points in each dimensiom
# and assumes Bruker ser file the object contains the entire serial file data
# points is a tuple of length that matches the number of dimension and each value is the 
# number of complex points . e.g. (1024, 32, 64) would be a 3D expeirment 
# with 1024 complex points in direct dimension (T3) and 32 complex points in T2 and 64 complex points in T1. 
# the order of this tuple is strictly important. Order should be (T4), (T3), (T2), T1


class LINData:

    def __init__(self, dataDir='.', serFile = 'ser', points=None, ddim=None, decim=None, dspfvs=None, grpdly=None):

        my_file = Path(dataDir+"/acqus")
        if my_file.is_file():
            acqusfile = open(dataDir+"/acqus", "r")
            for line in acqusfile:
                if "##$TD=" in line: 
                    (name, value) = line.split()
                    p0 = int(int(value)/2)

                if "##$DECIM=" in line:
                    (name, value) = line.split()
                    decf = float(value)
                    dec = int(decf)
                if "##$DSPFVS=" in line:
                    (name, value) = line.split()
                    dsp = int(float(value))
                if "##$GRPDLY=" in line:
                    (name, value) = line.split()
                    grp = float(value)
                
            acqusfile.close()
            p1 = None ; p2 = None
               
        my_file = Path(dataDir+"/acqu2s")
        if my_file.is_file():
            acqusfile = open(dataDir+"/acqu2s", "r")
            for line in acqusfile:
                if "##$TD=" in line: 
                    (name, value) = line.split()
                    p1 = int(int(value)/2)       
            acqusfile.close()
            p2 = None
            
        my_file = Path(dataDir+"/acqu3s")
        if my_file.is_file():
            acqusfile = open(dataDir+"/acqu3s", "r")
            for line in acqusfile:
                if "##$TD=" in line: 
                    (name, value) = line.split()
                    p2 = int(int(value)/2)       
            acqusfile.close()    
        
        if not points:
            if p0 and not p1 and not p2:
                points = [p0]
            if p0 and p1 and not p2:
                points = [p0, p1]
            if p0 and p1 and p2:
                points = [p0, p1, p2]

        if not decim:
            decim = dec
        if not dspfvs:
            dspfvs = dsp
        if not grpdly:
            if 'grp' in locals():
                grpdly = grp
            else:
                grpdly = self.dd2g(dspfvs,decim)
            
            
        print('Complex Points structure is: '+str(points))
        print('DECIM= '+str(decim)+' DSPFVS= '+str(dspfvs)+' GRPDLY= '+str(grpdly))
        #we need ot know the number of dimensions
        self.ndim = len(points)
        
        if ddim == None:
            self.ddim = ['t', 't', 't'] # ddim = domain of dimension. We assume when loaded we have 
                                       # time domain data. Unless specified otherwise
        else:
            self.ddim = ddim
        
        if ddim:
            if len(ddim) != len(points):
                raise ValueError("insanity: number of dimensions in 'points' and 'ddim' don't match")
            else:
                for i in range(len(ddim)):
                    if ddim[i] != 't' and ddim[i] != 'f':
                        print(ddim[i])
                        raise ValueError("dimension domains must be 'f' - frequency or 't' - time")
            
        #self.ddim = ddim
            
        # lets store the points
        self.points = points
        
        # typical bruker files are 4 bytes per data point
        # so keep in mind that file size is 4 * datasize
        numdatapoints = 1
        for p in points:
            numdatapoints *= p*2 # times 2 because we assume complex data
        self.datasize = numdatapoints
        
        # now lets load in the bruker serial file
        with open(dataDir+'/'+serFile, 'rb') as serialfile:
            self.lindata = np.frombuffer(serialfile.read(), dtype='>i4')
    
        # now reshape the data
        pointsnp = np.asarray(points)
        self.lindata = np.reshape(self.lindata, (pointsnp)*2, order='F')
        
        # TODO - set up some sort of sanity test
        # if len(self.lindata) == self.datasize:
        #    self.sane = True
        # else:
        #    self.sane = False
        
        # lets convert the data (only bruker right now)
        if decim and dspfvs:
            if grpdly:
                self.convertBruker(grpdly)
            else:
                grpdly = self.dd2g(dspfvs,decim)
                self.convertBruker(grpdly)
            
        elif grpdly and not decim and not dspfvs:
            self.convertBruker(grpdly)
            
        else:
            print("Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim")
 
    def dd2g(self, dspfvs, decim):
        dspDic = {
                10: {
                    2: 44.75,
                    3: 33.5,
                    4: 66.625,
                    6: 59.083333333333333,
                    8: 68.5625,
                    12: 60.375,
                    16: 69.53125,
                    24: 61.020833333333333,
                    32: 70.015625,
                    48: 61.34375,
                    64: 70.2578125,
                    96: 61.505208333333333,
                    128: 70.37890625,
                    192: 61.5859375,
                    256: 70.439453125,
                    384: 61.626302083333333,
                    512: 70.4697265625,
                    768: 61.646484375,
                    1024: 70.48486328125,
                    1536: 61.656575520833333,
                    2048: 70.492431640625,
                    },
                11: {
                    2: 46.,
                    3: 36.5,
                    4: 48.,
                    6: 50.166666666666667,
                    8: 53.25,
                    12: 69.5,
                    16: 72.25,
                    24: 70.166666666666667,
                    32: 72.75,
                    48: 70.5,
                    64: 73.,
                    96: 70.666666666666667,
                    128: 72.5,
                    192: 71.333333333333333,
                    256: 72.25,
                    384: 71.666666666666667,
                    512: 72.125,
                    768: 71.833333333333333,
                    1024: 72.0625,
                    1536: 71.916666666666667,
                    2048: 72.03125
                    },
                12: {
                    2: 46.,
                    3: 36.5,
                    4: 48.,
                    6: 50.166666666666667,
                    8: 53.25,
                    12: 69.5,
                    16: 71.625,
                    24: 70.166666666666667,
                    32: 72.125,
                    48: 70.5,
                    64: 72.375,
                    96: 70.666666666666667,
                    128: 72.5,
                    192: 71.333333333333333,
                    256: 72.25,
                    384: 71.666666666666667,
                    512: 72.125,
                    768: 71.833333333333333,
                    1024: 72.0625,
                    1536: 71.916666666666667,
                    2048: 72.03125
                    },
                13: {
                    2: 2.75,
                    3: 2.8333333333333333,
                    4: 2.875,
                    6: 2.9166666666666667,
                    8: 2.9375,
                    12: 2.9583333333333333,
                    16: 2.96875,
                    24: 2.9791666666666667,
                    32: 2.984375,
                    48: 2.9895833333333333,
                    64: 2.9921875,
                    96: 2.9947916666666667
                    }
                }

        return dspDic[dspfvs][decim]
        
        
    def convertBruker(self,
                 grpdly
                ):
        # edit the number of points in first dimension after Bruker filter removal
        self.points[0] = len(remove_bruker_filter(make_complex(self.lindata[:, 0, 0]), grpdly))
        # zero fill in a 3D matrix with complex zeros
        self.spectrum = np.zeros((self.points[0], self.points[1], self.points[2]), dtype='complex128')
        # load the data
        for ii in range(self.points[2]): # outer loop for third dimension points from dataFID
            for i in range(self.points[1]): # inner loop for second dimension points from dataFID
                fid = remove_bruker_filter(make_complex(self.lindata[:,i,ii]), grpdly)
                # fid = np.pad(fid, (0,self.procsize[0]-len(fid)), 'constant', constant_values=(0.+0.j))
                self.spectrum[0:len(fid),i,ii] = fid
        
        self.points[0] = len(fid) # set this so window function in direct dimension is correct after bruker filter
        self.procsize = self.points
        
    def zeroFill(self, zf=[1, 1, 1]):
        
        self.zf = zf
        
        # default zero filling of matrix to these sizes (next Fourier number)
        self.procsize = (2**(next_fourier_number(self.points[0])+zf[0]), 
            2**(next_fourier_number(self.points[1])+zf[1]), 
            2**(next_fourier_number(self.points[2])+zf[2]))
        
    def window(self, wp=[0.5, 0.5, 0.5]):
        self.wp = wp
        # at some point will allow more than just shifted sine bell
        
    def fpcorrection(self, c = [1.0, 1.0, 1.0]):
        self.c = c
        
    def phase(self, ph = [0, 0, 0]):
        self.ph = ph
        
    def transform(self, tf = ['f', 'f', 'f']):
        self.tf = tf
    
    def process3D(self,
                  zf=None,
                  wp=None,
                  c=None,
                  ph=None,
                  tf=None
                 ):
        
        if zf:
            self.zf = zf
            self.zeroFill(zf=zf)
            
        if wp:
            self.wp = wp
           
        if c:
            self.c = c
            
        if ph:
            self.ph = ph
            
        if tf:
            self.tf = tf
        
        # if some things aren't set, lets force defaults
        if not hasattr(self, 'wp'):
            self.wp = [0.5, 0.5, 0.5]
        
        if not hasattr(self, 'c'):
            self.c = [1.0, 1.0, 1.0]
            
        if not hasattr(self, 'ph'):
            self.ph = [0, 0, 0]
           
        if not hasattr(self, 'tf'):
            self.tf=['f', 'f', 'f']
            
        if not hasattr(self, 'zf'):
            self.zeroFill()
        
        # direct dimension phasing
        phcorr = np.exp(1.j*(self.ph[0]/180)*np.pi)
        
        # decide to transform or not
        if self.tf[0] == 'f' and self.ddim[0] == 't':
            # set the dimension state
            self.ddim[0] = 'f'
            # zero fill
            zfspectrum = np.zeros((self.procsize[0], self.spectrum.shape[1], self.spectrum.shape[2]), dtype='complex128')
            zfspectrum[0:self.spectrum.shape[0], 0:self.spectrum.shape[1], 0:self.spectrum.shape[2]] += self.spectrum[0:self.spectrum.shape[0], 0:self.spectrum.shape[1], 0:self.spectrum.shape[2]]
            self.spectrum = zfspectrum
            # fourier transform the direct dimension - we assume its simple DQD
            for ii in range(self.points[2]): # outer loop for third dimension points from dataFID
                for i in range(self.points[1]): # inner loop for second dimension points from dataFID
                    fid = self.spectrum[:,i,ii]
                    fid[0] = fid[0] * self.c[0] # first point correction
                    fid[0:self.points[0]] = fid[0:self.points[0]] * np.sin(( #window function
                        self.wp[0]*math.pi + (0.99-self.wp[0])*math.pi*np.arange(self.points[0])/self.points[0]))
                    fid = np.pad(fid, (0,self.procsize[0]-len(fid)), 'constant', constant_values=(0.+0.j))
                    self.spectrum[:,i,ii] = np.fft.fftshift(np.fft.fft(fid)*phcorr)[::-1]
        
        elif self.tf[0] == 't' and self.ddim[0] == 't':
            pass
            #for ii in range(self.points[2]*2): # outer loop for third dimension points from dataFID
            #    for i in range(self.points[1]*2): # inner loop for second dimension points from dataFID
            #        fid = self.spectrum[:,i,ii]
            #        fid = np.pad(fid, (0,self.procsize[0]-len(fid)), 'constant', constant_values=(0.+0.j))
            #        self.spectrum[:,i,ii] = fid*phcorr
        
            
            
                    
        
        # transpose data f3t2t1 -> t2f3t1
        self.spectrum = self.spectrum.transpose(1,0,2)
        # phase t2
        phcorr = np.exp(1.j*(self.ph[1]/180)*np.pi)
        
        # t2f3t1 -> f2f3t1 - this transform uses hypercomplex processing
        if self.tf[1] == 'f' and self.ddim[1] == 't':
            # set the dimension state
            self.ddim[1] = 'f'
            # zero fill
            zfspectrum = np.zeros((self.procsize[1], self.spectrum.shape[1], self.spectrum.shape[2]), dtype='complex128')
            zfspectrum[0:self.spectrum.shape[0], 0:self.spectrum.shape[1], 0:self.spectrum.shape[2]] += self.spectrum[0:self.spectrum.shape[0], 0:self.spectrum.shape[1], 0:self.spectrum.shape[2]]
            self.spectrum = zfspectrum
            
            for ii in range(self.spectrum.shape[2]): 
                for i in range(self.spectrum.shape[1]):
                    real = np.real(self.spectrum[::2,i,ii])
                    imag = np.real(self.spectrum[1::2,i,ii])
                    inter = np.ravel((real,imag), order='F')
                    fid = make_complex(inter)
                    fid[0] = fid[0] * self.c[1] # first point correction
                    #window function
                    fid = fid[0:self.points[1]] * np.sin((self.wp[1]*math.pi + (0.99-self.wp[1])*
                        math.pi*np.arange(self.points[1])/self.points[1]))
                    # because this throughts out half the data we need to pad it to 
                    # make it the same length as it was
                    fid = np.pad(fid, (0,self.procsize[1]-len(fid)), 'constant', constant_values=(0.+0.j))
                    self.spectrum[:,i,ii] = np.fft.fft(fid)*phcorr

        # f2f3t1 -> t1f3f2
        self.spectrum = self.spectrum.transpose(2, 1, 0)
        # indirect dimension phasing
        phcorr = np.exp(1.j*(self.ph[2]/180)*np.pi)

        # t1f3f2 -> f3f1f2 - this transform uses hypercomplex processing
        if self.tf[2] == 'f' and self.ddim[2] == 't':
            # set the dimension state
            self.ddim[2] = 'f'
            # zero fill
            zfspectrum = np.zeros((self.procsize[2], self.spectrum.shape[1], self.spectrum.shape[2]), dtype='complex128')
            zfspectrum[0:self.spectrum.shape[0], 0:self.spectrum.shape[1], 0:self.spectrum.shape[2]] += self.spectrum[0:self.spectrum.shape[0], 0:self.spectrum.shape[1], 0:self.spectrum.shape[2]]
            self.spectrum = zfspectrum
            
            for ii in range(self.spectrum.shape[2]): 
                for i in range(self.spectrum.shape[1]):
                    real = np.real(self.spectrum[::2,i,ii])
                    imag = np.real(self.spectrum[1::2,i,ii])
                    inter = np.ravel((real,imag), order='F')
                    fid = make_complex(inter)
                    fid[0] = fid[0] * self.c[2]  # first point correction
                    # window function
                    fid = fid[0:self.points[2]] * np.sin((self.wp[2]*math.pi + (0.99-self.wp[2])*
                        math.pi*np.arange(self.points[2])/self.points[2]))
                    # because this throws out half the data we need to pad it to
                    # make it the same length as it was
                    fid = np.pad(fid, (0,self.procsize[2]-len(fid)), 'constant', constant_values=(0.+0.j))
                    self.spectrum[:,i,ii] = np.fft.fft(fid)*phcorr

        # return spectrum to original ordering
        self.spectrum = self.spectrum.transpose(1, 2, 0)
