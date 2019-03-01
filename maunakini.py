import numpy as np
import math


from pathlib import Path
import os.path

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

        if 4 * 2**self.nusDimensions * self.nusPoints * self.pointsInDirectFid == 4*len(self.nusData):
            self.sane = True
        else:
            self.sane = False

        # reshape the data
        self.nusData = np.reshape(self.nusData, (self.pointsInDirectFid, 4, self.nusPoints), order='F')
        self.points = len(remove_bruker_filter(make_complex(np.copy(self.nusData[:, 0, 0])), grpdly))
        print('converted points: ', self.points)
        self.convertedNUSData = np.zeros((self.points, 4, self.nusPoints), dtype='complex128')

        # remove bruker filter
        for ii in range(self.nusPoints):  #
            for i in range(4):
                fid = remove_bruker_filter(make_complex(np.copy(self.nusData[:, i, ii])), grpdly)
                self.convertedNUSData[0:len(fid), i, ii] = fid

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

    def order_data(self):

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

    def __getitem__(self, sliced):
        return self.convertedNUSData.T[sliced]


# define class for Linear bruker data - this includes the number of points in each dimensiom
# and assumes Bruker ser file the object contains the entire serial file data
# points is a tuple of length that matches the number of dimension and each value is the 
# number of complex points . e.g. (1024, 32, 64) would be a 3D expeirment 
# with 1024 complex points in direct dimension (T3) and 32 complex points in T2 and 64 complex points in T1. 
# the order of this tuple is strictly important. Order should be (T4), (T3), (T2), T1


class LINData:

    def __init__(self, data_dir='.', ser_file='ser', points=None,
                 dim_status=None, decim=None, dspfvs=None, grpdly=None,
                 ):

        my_file = Path(data_dir+"/acqus")
        if my_file.is_file():
            acqusfile = open(data_dir+"/acqus", "r")
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
               
        my_file = Path(data_dir+"/acqu2s")
        if my_file.is_file():
            acqusfile = open(data_dir+"/acqu2s", "r")
            for line in acqusfile:
                if "##$TD=" in line: 
                    (name, value) = line.split()
                    p1 = int(int(value)/2)       
            acqusfile.close()
            p2 = None
            
        my_file = Path(data_dir+"/acqu3s")
        if my_file.is_file():
            acqusfile = open(data_dir+"/acqu3s", "r")
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
                grpdly = dd2g(dspfvs, decim)

        print('Complex Points structure is: '+str(points))
        print('DECIM= '+str(decim)+' DSPFVS= '+str(dspfvs)+' GRPDLY= '+str(grpdly))
        # we need ot know the number of dimensions
        self.num_dim = len(points)
        
        if dim_status is None:
            self.dim_status = ['t', 't', 't']  # dim status: is data in f or t domain. We assume all t at first
        else:
            self.dim_status = dim_status
        
        if dim_status:
            if len(dim_status) != len(points):
                raise ValueError("insanity: number of dimensions in 'points' and 'dim_status' don't match")
            else:
                for i in range(len(dim_status)):
                    if dim_status[i] != 't' and dim_status[i] != 'f':
                        print(dim_status[i])
                        raise ValueError("dimension domains must be 'f' - frequency or 't' - time")
            
        # lets store the points
        self.points = points

        self.process_size = self.points

        # typical bruker files are 4 bytes per data point
        # so keep in mind that file size is 4 * datasize
        num_data_points = 1
        for p in points:
            num_data_points *= p*2 # times 2 because we assume complex data
        self.data_size = num_data_points
        
        # now lets load in the bruker serial file
        with open(data_dir+'/'+ser_file, 'rb') as serial_file:
            self.raw_data = np.frombuffer(serial_file.read(), dtype='<i4')
    
        # now reshape the data
        points_np = np.asarray(points)
        self.raw_data = np.reshape(self.raw_data, points_np*2, order='F')
        
        # TODO - set up some sort of sanity test
        # if len(self.lindata) == self.datasize:
        #    self.sane = True
        # else:
        #    self.sane = False
        self.converted_data = np.zeros((self.points[0], self.points[1], self.points[2]), dtype='complex128')

        # lets convert the data (only bruker right now)
        if decim and dspfvs:
            if grpdly:
                self.convert_bruker(grpdly)
            else:
                grpdly = dd2g(dspfvs, decim)
                self.convert_bruker(grpdly)
            
        elif grpdly and not decim and not dspfvs:
            self.convert_bruker(grpdly)
            
        else:
            print("Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim")

    def convert_bruker(self, grpdly):

        # edit the number of points in first dimension after Bruker filter removal
        self.points[0] = len(remove_bruker_filter(make_complex(self.raw_data[:, 0, 0]), grpdly))

        # zero fill in a 3D matrix with complex zeros

        # load the data
        for ii in range(self.points[2]): # outer loop for third dimension points from dataFID
            for i in range(self.points[1]): # inner loop for second dimension points from dataFID
                fid = remove_bruker_filter(make_complex(self.raw_data[:, i, ii]), grpdly)
                self.converted_data[0:len(fid), i, ii] = fid
        
        self.process_size = self.points
        self.converted_data = self.converted_data[
                              0:self.points[0],
                              0:self.points[1],
                              0:self.points[2],
                              ]
        self.raw_data = self.converted_data  # clean up memory a little


class LINProc:

    def __init__(self,
                 lin_data,  # this should be of type LINData
                 zero_fills=(1, 1, 1),  # this just means to next Fourier number - no real zero fill
                 windows=(0.5, 0.5, 0.5),
                 fp_corrections=(1.0, 0.5, 0.5),
                 phases=(0, 0, 0),
                 tf=('f', 'f', 'f'),  # the state we want finally
                 ):

        self.windows = list(windows)
        self.fp_corrections = list(fp_corrections)
        self.phases = list(phases)
        self.tf = list(tf)
        self.LIN_data = lin_data
        self.zero_fills = list(zero_fills)

        # final spectrum size will be this size - includes the zero fills
        self.ft_points = [2 ** (next_fourier_number(lin_data.points[0]) + self.zero_fills[0]),
                          2 ** (next_fourier_number(lin_data.points[1]) + self.zero_fills[1]),
                          2 ** (next_fourier_number(lin_data.points[2]) + self.zero_fills[2]),
                          ]

        self.spectrum = np.zeros((self.ft_points[0],
                                  self.ft_points[1],
                                  self.ft_points[2]),
                                 dtype='complex128')

        # unprocessed data is this size
        self.data_points = list(self.LIN_data.converted_data.shape)
    
    def process3d(self):
        
        # direct dimension phasing
        phase_correction = [
            np.exp(1.j * (self.phases[0] / 180) * np.pi),
            np.exp(1.j * (self.phases[1] / 180) * np.pi),
            np.exp(1.j * (self.phases[2] / 180) * np.pi),
        ]
        
        # decide to transform or not
        if self.tf[0] == 'f' and self.LIN_data.dim_status[0] == 't':

            # set the dimension state
            self.LIN_data.dim_status[0] = 'f'

            # create a zeroed spectrum with zero fill
            zf_spectrum = self.spectrum

            # add the LINData spectrum to zf_spectrum but only to the size of LINData
            # the rest of zf_spectrum stays as zeros. i.e. zero filled!
            zf_spectrum[0:self.data_points[0],
                        0:self.data_points[1],
                        0:self.data_points[2],
                        ] += self.LIN_data.converted_data

            self.spectrum = zf_spectrum

            window = np.sin((
                        self.windows[0]*math.pi +
                        (0.99-self.windows[0]) *
                        math.pi*np.arange(self.data_points[0]) /
                        self.data_points[0]))

            # fourier transform the direct dimension - we assume its simple DQD
            # note, we loop to data_points, not length of zf_spectrum (self.spectrum).
            # So we don't do FT on lines of zeroes.
            for ii in range(self.data_points[2]):  # outer loop for third dimension points from dataFID
                for i in range(self.data_points[1]):  # inner loop for second dimension points from dataFID

                    # grab the fid we need
                    fid = self.spectrum[:, i, ii]

                    # first point correction
                    fid[0] = fid[0] * self.fp_corrections[0]

                    # window function
                    fid[0:self.data_points[0]] = fid[0:self.data_points[0]] * window

                    self.spectrum[:, i, ii] = np.fft.fftshift(np.fft.fft(fid)*phase_correction[0])[::-1]

            self.data_points[0] = self.spectrum.shape[0]

        # transpose data f3t2t1 -> t2f3t1
        self.spectrum = self.spectrum.transpose(1, 0, 2)

        # t2f3t1 -> f2f3t1 - this transform uses hypercomplex processing
        if self.tf[1] == 'f' and self.LIN_data.dim_status[1] == 't':
            # set the dimension state
            self.LIN_data.dim_status[1] = 'f'

            window = np.sin((
                    self.windows[1] * math.pi +
                    (0.99 - self.windows[1]) *
                    math.pi * np.arange(self.data_points[1]) /
                    self.data_points[1]))
            print(self.data_points[1])
            print(self.spectrum.shape)
            for ii in range(self.data_points[2]):
                for i in range(self.data_points[0]):
                    real = np.real(self.spectrum[::2, i, ii])
                    imag = np.real(self.spectrum[1::2, i, ii])
                    inter = np.ravel((real, imag), order='F')
                    fid = make_complex(inter)
                    #print(len(fid), len(real))
                    fid[0] = fid[0] * self.fp_corrections[1]  # first point correction
                    # window function
                    fid[0:self.data_points[1]] = fid[0:self.data_points[1]] * window

                    self.spectrum[0:len(fid), i, ii] = \
                        np.fft.fftshift(np.fft.fft(fid)*phase_correction[1])[::-1]

        # f2f3t1 -> t1f3f2
        self.spectrum = self.spectrum.transpose(2, 1, 0)

        # t1f3f2 -> f3f1f2 - this transform uses hypercomplex processing
        if self.tf[2] == 'f' and self.LIN_data.dim_status[2] == 't':
            # set the dimension state
            self.LIN_data.dim_status[2] = 'f'

            window = np.sin((
                    self.windows[2] * math.pi +
                    (0.99 - self.windows[2]) *
                    math.pi * np.arange(self.data_points[2]) /
                    self.data_points[2]))
            for ii in range(self.data_points[1]):
                for i in range(self.data_points[0]):
                    real = np.real(self.spectrum[::2, i, ii])
                    imag = np.real(self.spectrum[1::2, i, ii])
                    inter = np.ravel((real, imag), order='F')
                    fid = make_complex(inter)
                    fid[0] = fid[0] * self.fp_corrections[2]  # first point correction
                    # window function
                    fid[0:self.data_points[2]] = fid[0:self.data_points[2]] * window

                    self.spectrum[0:len(fid), i, ii] = np.fft.fftshift(np.fft.fft(fid)*phase_correction[2])[::-1]

        # return spectrum to original ordering
        self.spectrum = self.spectrum.transpose(1, 2, 0)


class LINData2D:

    def __init__(self, data_dir='.', ser_file='ser', points=None,
                 dim_status=None, decim=None, dspfvs=None, grpdly=None,
                 ):

        self.ac1 = os.path.join(data_dir, 'acqus')
        self.ac2 = os.path.join(data_dir, 'acqu2s')
        self.ser = os.path.join(data_dir, 'ser')
        self.pp = os.path.join(data_dir, 'pulseprogram')
        self.ser = os.path.join(data_dir, ser_file)
        self.dir = data_dir
        self.acq = [0, 0]  # acquisition modes start as undefined

        # dictionary of acquisition modes for Bruker
        self.acqDict = {0: 'undefined',
                        1: 'qf',
                        2: 'qsec',
                        3: 'tppi',
                        4: 'states',
                        5: 'states-tppi',
                        6: 'echo-antiecho',
                        }

        # check if we are a Bruker 2D data set
        if (os.path.isfile(self.ac1) and
                os.path.isfile(self.ac2) and
                os.path.isfile(self.ser) and
                os.path.isfile(self.pp)):
            self.valid = True

        else:
            self.valid = False
            print('Data Directory does not seem to contain Bruker 2D Data')

        p0 = p1 = 0  # we'll find these in the files
        dec = dsp = grp = 0  # we'll find these in the files

        acqusfile = open(self.ac1, "r")
        for line in acqusfile:
            if "##$TD=" in line:
                (name, value) = line.split()
                p0 = int(value)
            if "##$DECIM=" in line:
                (name, value) = line.split()
                dec = int(value)
            if "##$DSPFVS=" in line:
                (name, value) = line.split()
                dsp = int(value)
            if "##$GRPDLY=" in line:
                (name, value) = line.split()
                grp = float(value)

            self.acq[0] = 0  # doesnt matter we assume DQD for direct anyway

        acqusfile.close()

        acqusfile = open(self.ac2, "r")
        for line in acqusfile:
            if "##$TD=" in line:
                (name, value) = line.split()
                p1 = int(value)

            if "##$FnMODE=" in line:
                (name, value) = line.split()
                self.acq[1] = int(value)

        acqusfile.close()

        if p0 and p1:
            points = [p0, p1]
        else:
            print("problem with detecting number of points in data")
            self.valid = False

        if dec != 0:
            decim = dec
        if dsp != 0:
            dspfvs = dsp
        if grp:
            grpdly = grp
        elif dec != 0 and dsp != 0:
            grpdly = dd2g(dspfvs, decim)
        else:
            print("problem with detecting / determining grpdly - needed for Bruker conversion")
            self.valid = False

        print('Data Points structure is: ' + str(points))
        print('DECIM= ' + str(decim) + ' DSPFVS= ' + str(dspfvs) + ' GRPDLY= ' + str(grpdly))

        if dim_status is None:
            self.dim_status = ['t', 't']  # dim status: is data in f or t domain. We assume all t at first
        else:
            self.dim_status = dim_status

        if dim_status:
            if len(dim_status) != len(points):
                raise ValueError("insanity: number of dimensions in 'points' and 'dim_status' don't match")
            else:
                for i in range(len(dim_status)):
                    if dim_status[i] != 't' and dim_status[i] != 'f':
                        print(dim_status[i])
                        raise ValueError("dimension domains must be 'f' - frequency or 't' - time")

        # lets store the points
        self.points = points

        # now lets load in the bruker serial file
        with open(self.ser, 'rb') as serial_file:
            self.raw_data = np.frombuffer(serial_file.read(), dtype='<i4')

        # now reshape the data
        self.raw_data = np.reshape(self.raw_data, np.asarray(self.points), order='F')

        # TODO - set up some sort of sanity test
        # if len(self.lindata) == self.datasize:
        #    self.sane = True
        # else:
        #    self.sane = False

        # converted data is going to be of complex numbers. So first dimension will at least be halved
        # since is is acquired as DQD or points = R+I. So complex numbers = R = I. But indirect
        # dimension is still a mix of the 2 experiments with 90 phase shift.
        self.converted_data = np.zeros((int(self.points[0]/2), self.points[1]), dtype='complex128')

        # lets convert the data
        if decim and dspfvs:
            if grpdly:
                self.convert_bruker_2d(grpdly)
            else:
                grpdly = dd2g(dspfvs, decim)
                self.convert_bruker_2d(grpdly)

        elif grpdly and not decim and not dspfvs:
            self.convert_bruker_2d(grpdly)

        else:
            print("Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim")

        print('Converted Data Points structure is:', self.points)

        self.phases = (0, 0)
        self.fp_corrections = (0.5, 0.5)
        self.windows = ('sb', 'sb')
        self.windows_p = (0.5, 0.5)
        self.zero_fill = (1.0, 1.0)

        self.processed_data = []  # this will be filled out in proc method
        self.ft_points = []

    def convert_bruker_2d(self, grpdly):

        # edit the number of points in first dimension after Bruker filter removal
        # we now count points in complex numbers as well
        self.points[0] = len(remove_bruker_filter(make_complex(self.raw_data[:, 0]), grpdly))

        # convert the data
        for i in range(self.points[1]):  # inner loop for second dimension points from dataFID
            fid = remove_bruker_filter(make_complex(self.raw_data[:, i]), grpdly)
            self.converted_data[0:len(fid), i] = fid

        self.converted_data = self.converted_data[
                              0:self.points[0],
                              0:self.points[1],
                              ]

        if self.acq[1] == 6:  # Rance Kay Processing needed
            print('Echo-AntiEcho Detected in T1 - dealing with it...')
            for i in range(0, self.points[1], 2):
                a = self.converted_data[:, i]
                b = self.converted_data[:, i+1]
                c = a + b
                d = a - b
                self.converted_data[:, i] = c * np.exp(1.j * (90 / 180) * np.pi)
                self.converted_data[:, i+1] = d * np.exp(1.j * (180 / 180) * np.pi)

        self.raw_data = self.converted_data  # clean up memory a little

    def proc_t2(self, phase=0, c=1.0, window='sb', window_p=0.5):

        self.processed_data[0, :] = self.processed_data[0, :] * c

        for i in range(self.ft_points[1]):
            fid = self.processed_data[:, i]
            fid = fid * self.window_function(points=len(fid),
                                             window=window,
                                             window_p=window_p,
                                             )
            self.processed_data[0:len(fid), i] = fid
            self.processed_data[:, i] = np.fft.fftshift(
                np.fft.fft(self.processed_data[:, i] * np.exp(1.j * (phase / 180) * np.pi)))[::-1]

    def proc_t1(self, phase=0, c=1.0, window='sb', window_p=0.5):

        self.processed_data[:, 0] = self.processed_data[:, 0] * c
        self.processed_data[:, 1] = self.processed_data[:, 1] * c

        for i in range(self.ft_points[0]):
            fid_r = np.real(self.processed_data[i, ::2])
            fid_i = np.real(self.processed_data[i, 1::2])
            fid = np.ravel((fid_r, fid_i), order='F')
            fid = make_complex(fid)
            fid = fid * self.window_function(points=len(fid),
                                             window=window,
                                             window_p=window_p
                                             )

            self.processed_data[i, 0:len(fid)] = fid
            self.processed_data[i, len(fid):] = np.zeros(self.ft_points[1]-len(fid))

            if self.acq[1] != 5 or self.acq[1] != 5:
                self.processed_data[i, :] = np.fft.fftshift(
                    np.fft.fft(self.processed_data[i, :] * np.exp(1.j * (phase / 180) * np.pi)))[::-1]

            elif self.acq[1] == 5 or self.acq[1] == 3:  # states tppi or tppi - don't fftshift
                self.processed_data[i, :] = np.fft.fft(
                    self.processed_data[i, :] * np.exp(1.j * (phase / 180) * np.pi))[::-1]

    def proc(self, phases=(0, 0),
             fp_corrections=(0.5, 0.5),
             windows=('sb', 'sb'),
             windows_p=(0.5, 0.5),
             zero_fill=(1.0, 1.0),
             ):

        t1_ac_mode = int(self.acq[1])
        if t1_ac_mode >= 3 or t1_ac_mode <= 6:  # hypercomplex data. T1 points is really half
            points_t2 = int(self.points[1] / 2)
        else:
            points_t2 = self.points[1]

        self.ft_points = (int(2 ** (next_fourier_number(self.points[0]) + zero_fill[0])),
                          int(2 ** (next_fourier_number(points_t2) + zero_fill[1])),
                          )
        print(self.ft_points)
        self.processed_data = np.zeros(self.ft_points, dtype='complex128')

        self.processed_data[0:self.points[0], 0:self.points[1]] = self.converted_data

        self.proc_t2(phase=phases[0],
                     c=fp_corrections[0],
                     window=windows[0],
                     window_p=windows_p[0],
                     # ft_points=ft_points[0],
                     )

        self.proc_t1(phase=phases[1],
                     c=fp_corrections[1],
                     window=windows[1],
                     window_p=windows_p[1],
                     # ft_points=ft_points[1],
                     )

    @staticmethod
    def window_function(points=0, window='sb', window_p=0.5):
        if window == 'sb':
            w = np.sin((
                window_p * math.pi +
                (0.99 - window_p) *
                math.pi * np.arange(points) /
                points))
        else:
            w = np.ones(points)

        return w
