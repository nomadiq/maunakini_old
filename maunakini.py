import numpy as np
import scipy.signal as signal
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


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


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

# ------------------------------------- #
#               Classes                 #
# ------------------------------------- #


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

            if "##$BYTORDA=" in line:
                (name, value) = line.split()
                self.byte_order = float(value)

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
            if self.byte_order == 0:
                self.raw_data = np.frombuffer(serial_file.read(), dtype='<i4')
            elif self.byte_order == 1:
                self.raw_data = np.frombuffer(serial_file.read(), dtype='>i4')

        # now reshape the data
        self.raw_data = np.reshape(self.raw_data, np.asarray(self.points), order='F')

        # TODO - set up some sort of sanity test

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
            fid = fid * window_function(points=len(fid),
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
            fid = fid * window_function(points=len(fid),
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
                     )

        self.proc_t1(phase=phases[1],
                     c=fp_corrections[1],
                     window=windows[1],
                     window_p=windows_p[1],
                     )


class LINData3D:

    def __init__(self, data_dir='.', ser_file='ser', points=None,
                 dim_status=None, decim=None, dspfvs=None, grpdly=None,
                 ):

        self.ac1 = os.path.join(data_dir, 'acqus')
        self.ac2 = os.path.join(data_dir, 'acqu2s')
        self.ac3 = os.path.join(data_dir, 'acqu3s')
        self.ser = os.path.join(data_dir, 'ser')
        self.pp = os.path.join(data_dir, 'pulseprogram')
        self.ser = os.path.join(data_dir, ser_file)
        self.dir = data_dir
        self.acq = [0, 0, 0]  # acquisition modes start as undefined

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
                os.path.isfile(self.ac3) and
                os.path.isfile(self.ser) and
                os.path.isfile(self.pp)):
            self.valid = True

        else:
            self.valid = False
            print('Data Directory does not seem to contain Bruker 2D Data')

        p0 = p1 = 0  # we'll find these in the files
        dec = dsp = grp = 0  # we'll find these in the files

        # read the first dimension details
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

            if "##$BYTORDA=" in line:
                (name, value) = line.split()
                self.byte_order = float(value)

            self.acq[0] = 0  # doesnt matter we assume DQD for direct anyway

        acqusfile.close()

        # read second dimension details
        acqusfile = open(self.ac2, "r")
        for line in acqusfile:
            if "##$TD=" in line:
                (name, value) = line.split()
                p1 = int(value)

            if "##$FnMODE=" in line:
                (name, value) = line.split()
                self.acq[1] = int(value)

        acqusfile.close()

        # read third dimension details
        acqusfile = open(self.ac3, "r")
        for line in acqusfile:
            if "##$TD=" in line:
                (name, value) = line.split()
                p2 = int(value)

            if "##$FnMODE=" in line:
                (name, value) = line.split()
                self.acq[2] = int(value)

        acqusfile.close()

        if p0 and p1 and p2:  # we got # points for all three dimensions
            points = [p0, p1, p2]
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

        # lets store the points to the class instance
        self.points = points

        # now lets load in the bruker serial file
        with open(self.ser, 'rb') as serial_file:
            if self.byte_order == 0:
                self.raw_data = np.frombuffer(serial_file.read(), dtype='<i4')
            elif self.byte_order == 1:
                self.raw_data = np.frombuffer(serial_file.read(), dtype='>i4')

        # now reshape the data
        self.raw_data = np.reshape(self.raw_data, np.asarray(self.points), order='F')

        # TODO - set up some sort of sanity test

        self.converted_data = np.zeros((int(self.points[0]/2), self.points[1], self.points[2]), dtype='complex128')

        # lets convert the data
        if decim and dspfvs:
            if grpdly:
                self.convert_bruker_3d(grpdly)
            else:
                grpdly = dd2g(dspfvs, decim)
                self.convert_bruker_3d(grpdly)

        elif grpdly and not decim and not dspfvs:
            self.convert_bruker_3d(grpdly)

        else:
            print("Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim")

        print('Converted Data Points structure is:', self.points)

        self.phases = (0, 0, 0)
        self.fp_corrections = (0.5, 0.5, 0.5)
        self.windows = ('sb', 'sb', 'sb')
        self.windows_p = (0.5, 0.5, 0.5)
        self.zero_fill = (1.0, 1.0, 1.0)

        self.processed_data = []  # this will be filled out in proc method
        self.ft_points = []

    def convert_bruker_3d(self, grpdly):

        # edit the number of points in first dimension after Bruker filter removal
        # we now count points in complex numbers as well
        self.points[0] = len(remove_bruker_filter(make_complex(self.raw_data[:, 0, 0]), grpdly))
        for ii in range(self.points[2]):  # outer loop for third dimension points from dataFID
            for i in range(self.points[1]):  # inner loop for second dimension points from dataFID
                fid = remove_bruker_filter(make_complex(self.raw_data[:, i, ii]), grpdly)
                self.converted_data[0:len(fid), i, ii] = fid

        self.converted_data = self.converted_data[
                              0:self.points[0],
                              0:self.points[1],
                              0:self.points[2],
                              ]
        self.raw_data = self.converted_data  # clean up memory a little

        if self.acq[1] == 6:  # Rance Kay Processing needed
            print('Echo-AntiEcho Detected in T2 - dealing with it...')
            for i in range(0, self.points[1], 2):
                for ii in range(0, self.points[2]):
                    a = self.converted_data[:, i, ii]
                    b = self.converted_data[:, i+1, ii]
                    c = a + b
                    d = a - b
                    self.converted_data[:, i, ii] = c * np.exp(1.j * (90 / 180) * np.pi)
                    self.converted_data[:, i+1, ii] = d * np.exp(1.j * (180 / 180) * np.pi)

        if self.acq[2] == 6:  # Rance Kay Processing needed
            print('Echo-AntiEcho Detected in T1 - dealing with it...')
            for i in range(0, self.points[2], 2):
                for ii in range(0, self.points[1]):
                    a = self.converted_data[:, ii, i]
                    b = self.converted_data[:, ii, i+1]
                    c = a + b
                    d = a - b
                    self.converted_data[:, ii, i] = c * np.exp(1.j * (90 / 180) * np.pi)
                    self.converted_data[:, i1, i+1] = d * np.exp(1.j * (180 / 180) * np.pi)

        self.raw_data = self.converted_data  # clean up memory a little

    def proc_t3(self, phase=0, t3_ss=None, c=1.0, window='sb', window_p=0.5):

        self.processed_data[0, :, :] = self.processed_data[0, :, :] * c
        window = window_function(points=self.points[0],
                                 window=window,
                                 window_p=window_p,
                                 )
        for i in range(self.ft_points[2]):
            for ii in range(self.ft_points[1]):
                fid = self.processed_data[:, ii, i]

                if t3_ss == 'poly':
                    co_ef = np.polynomial.polynomial.polyfit(np.arange(len(fid)),  fid,  5)
                    polyline = 0
                    time_points = np.arange(len(fid))
                    for iii in range(len(co_ef)):
                        # print(co_ef)
                        polyline += co_ef[iii] * time_points ** iii  # add the i'th order polynomial to polyline

                    fid = fid - polyline

                elif t3_ss == 'butter':
                    fid = butter_highpass_filter(fid, 0.01, 0.05, order=1)

                fid[0:len(window)] = fid[0:len(window)] * window

                self.processed_data[0:len(fid), ii, i] = fid
                self.processed_data[:, ii, i] = np.fft.fftshift(
                    np.fft.fft(self.processed_data[:, ii, i] * np.exp(1.j * (phase / 180) * np.pi)))[::-1]

    def proc_t2(self, phase=0, c=1.0, window='sb', window_p=0.5):

        self.processed_data[:, 0, :] = self.processed_data[:, 0, :] * c
        self.processed_data[:, 1, :] = self.processed_data[:, 1, :] * c

        window = window_function(points=self.points[1]/2,
                                 window=window,
                                 window_p=window_p
                                 )
        for i in range(self.ft_points[2]):
            for ii in range(self.ft_points[0]):
                fid_r = np.real(self.processed_data[ii, ::2, i])
                fid_i = np.real(self.processed_data[ii, 1::2, i])
                fid = np.ravel((fid_r, fid_i), order='F')
                fid = make_complex(fid)
                fid[0:int(self.points[1]/2)] = fid[0:int(self.points[1]/2)] * window

                self.processed_data[ii, 0:len(fid), i] = fid
                self.processed_data[ii, len(fid):, i] = np.zeros(self.ft_points[1]-len(fid))

                if self.acq[1] != 5 or self.acq[1] != 5:
                    self.processed_data[ii, :, i] = np.fft.fftshift(
                        np.fft.fft(self.processed_data[ii, :, i] * np.exp(1.j * (phase / 180) * np.pi)))[::-1]

                elif self.acq[1] == 5 or self.acq[1] == 3:  # states tppi or tppi - don't fftshift
                    self.processed_data[ii, :, i] = np.fft.fft(
                        self.processed_data[ii, :, i] * np.exp(1.j * (phase / 180) * np.pi))[::-1]

    def proc_t1(self, phase=0, c=1.0, window='sb', window_p=0.5):

        self.processed_data[:, :, 0] = self.processed_data[:, :, 0] * c
        self.processed_data[:, :, 1] = self.processed_data[:, :, 1] * c
        window = window_function(points=self.points[2]/2,
                                 window=window,
                                 window_p=window_p
                                 )
        for i in range(self.ft_points[1]):
            for ii in range(self.ft_points[0]):
                fid_r = np.real(self.processed_data[ii, i, ::2])
                fid_i = np.real(self.processed_data[ii, i, 1::2])
                fid = np.ravel((fid_r, fid_i), order='F')
                fid = make_complex(fid)
                fid[0:int(self.points[2] / 2)] = fid[0:int(self.points[2] / 2)] * window
                self.processed_data[ii, i, 0:len(fid)] = fid
                self.processed_data[ii, i, len(fid):] = np.zeros(self.ft_points[2]-len(fid))

                if self.acq[2] == 5 or self.acq[2] == 3:
                    self.processed_data[ii, i, :] = np.fft.fft(
                        self.processed_data[ii, i, :] * np.exp(1.j * (phase / 180) * np.pi))[::-1]

                else:  # states tppi or tppi - don't fftshift
                    self.processed_data[ii, i, :] = np.fft.fftshift(np.fft.fft(
                        self.processed_data[ii, i, :] * np.exp(1.j * (phase / 180) * np.pi)))[::-1]

    def proc(self, phases=(0, 0, 0),
             t3_ss=None,
             fp_corrections=(0.5, 0.5, 0.5),
             windows=('sb', 'sb', 'sb'),
             windows_p=(0.5, 0.5, 0.5),
             zero_fill=(1.0, 1.0, 1.0),
             ):

        t1_ac_mode = int(self.acq[1])
        if t1_ac_mode >= 3 or t1_ac_mode <= 6:  # hypercomplex data. T2 points is really half
            points_t2 = int(self.points[1] / 2)
        else:
            points_t2 = self.points[1]

        t1_ac_mode = int(self.acq[2])
        if t1_ac_mode >= 3 or t1_ac_mode <= 6:  # hypercomplex data. T1 points is really half
            points_t1 = int(self.points[2] / 2)
        else:
            points_t1 = self.points[2]

        self.ft_points = (int(2 ** (next_fourier_number(self.points[0]) + zero_fill[0])),
                          int(2 ** (next_fourier_number(points_t2) + zero_fill[1])),
                          int(2 ** (next_fourier_number(points_t1) + zero_fill[2])),
                          )
        print(self.ft_points)
        self.processed_data = np.zeros(self.ft_points, dtype='complex128')

        self.processed_data[0:self.points[0], 0:self.points[1], 0:self.points[2]] = self.converted_data[0:self.points[0], 0:self.points[1], 0:self.points[2]]

        print('Processing t3')
        self.proc_t3(phase=phases[0],
                     t3_ss=t3_ss,
                     c=fp_corrections[0],
                     window=windows[0],
                     window_p=windows_p[0],
                     )
        print('Processing t2')
        self.proc_t2(phase=phases[1],
                     c=fp_corrections[1],
                     window=windows[1],
                     window_p=windows_p[1],
                     )
        print('Processing t1')
        self.proc_t1(phase=phases[2],
                     c=fp_corrections[2],
                     window=windows[2],
                     window_p=windows_p[2],
                     )

