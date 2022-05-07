import collections
import os
import random
import tempfile
from typing import Union
import numpy as np
import os
import sys
import soundfile as sf
# cur_path=os.path.abspath(os.path.dirname(__file__))
# print(cur_path)
# sys.path.insert(0, cur_path+"/..")
import scipy.io as sio
import librosa
import scipy
import matplotlib.pyplot as plt
import json
import pickle
import torch

import torchaudio as ta
from scipy import signal
import re
from src.utils.active_act import SSfilter,SAD

__all__ = [
    'TextLineDataset',
    'SilentDataset',
    'ZipDataset',
    'TestSilentDataset'
]




def ceshi2(X, fs):            
    b1, a1 = signal.iirnotch(50, 30, fs) # 50 Hz 陷波器参数  
    b2, a2 = signal.iirnotch(150, 30, fs)  
    b3, a3 = signal.iirnotch(250, 30, fs)   
    b4, a4 = signal.iirnotch(350, 30, fs)  
    b5, a5 = signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], 'bandpass') 
    # 10-400 Hz 巴特沃斯带通滤波器参数          
    # # b2, a2 = signal.butter(4, [10, 400], 'bandpass', fs = 1000)         
    X = signal.filtfilt(b1, a1, X, axis=1)          
    X = signal.filtfilt(b2, a2, X, axis=1)          
    X = signal.filtfilt(b3, a3, X, axis=1) 
    X = signal.filtfilt(b4, a4, X, axis=1)   
    X = signal.filtfilt(b5, a5, X, axis=1) 

    return X



class Record(object):
    """
    ```Record``` is one sample of a ```Dataset```. It has three attributions: ```data```, ```key``` and ```n_fields```.

    ```data``` is the actual data format of one sample. It can be a single field or more.
    ```key``` is used in bucketing, the larger of which means the size of the data.
    ```
    """
    __slots__ = ("fields", "index")

    def __init__(self, *fields, index):

        self.fields = fields
        self.index = index

    @property
    def n_fields(self):
        return len(self.fields)


def zip_records(*records: Record):
    """
    Combine several records into one single record. The key of the new record is the
    maximum of previous keys.
    """
    new_fields = ()
    indices = []

    for r in records:
        new_fields += r.fields
        indices.append(r.index)

    return Record(*new_fields, index=max(indices))


def shuffle(*path):

    f_handles = [open(p, encoding='utf8') for p in path]

    # Read all the data
    lines = []
    for l in f_handles[0]:
        line = [l.strip()] + [ff.readline().strip() for ff in f_handles[1:]]
        lines.append(line)

    # close file handles
    [f.close() for f in f_handles]

    # random shuffle the data
    print('Shuffling data...')
    random.shuffle(lines)
    print('Done.')

    # Set up temp files
    f_handles = []
    for p in path:
        _, filename = os.path.split(p)
        f_handles.append(tempfile.TemporaryFile(prefix=filename + '.shuf', dir="/tmp/", mode="a+"))

    for line in lines:
        for ii, f in enumerate(f_handles):
            print(line[ii], file=f)

    # release memory
    lines.clear()

    # Reset file handles
    [f.seek(0) for f in f_handles]

    return tuple(f_handles)


class Dataset(object):
    """
    In ```Dataset``` object, you can define how to read samples from different formats of
    raw data, and how to organize these samples. Each time the ```Dataset``` return one record.

    There are some things you need to override:
        - In ```n_fields``` you should define how many fields in one sample.
        - In ```__len__``` you should define the capacity of your dataset.
        - In ```_data_iter``` you should define how to read your data, using shuffle or not.
        - In ```_apply``` you should define how to transform your raw data into some kind of format that can be
        computation-friendly. Must wrap the return value in a ```Record```， or return a ```None``` if this sample
        should not be output.
    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def data_path(self):
        raise NotImplementedError

    @property
    def n_fields(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _apply(self, *lines) -> Union[Record, None]:
        """ Do some processing on the raw input of the dataset.

        Return ```None``` when you don't want to output this line.

        Args:
            lines: A tuple representing one line of the dataset, where ```len(lines) == self.n_fields```

        Returns:
            A tuple representing the processed output of one line, whose length equals ```self.n_fields```
        """
        raise NotImplementedError

    def _data_iter(self, shuffle):

        if shuffle:
            return shuffle(self.data_path)
        else:
            return open(self.data_path)

    def data_iter(self, shuffle=False):

        f_handles = self._data_iter(shuffle=shuffle)

        if not isinstance(f_handles, collections.Sequence):
            f_handles = [f_handles]

        for lines in zip(*f_handles):

            record = self._apply(*lines)

            if record is not None:
                yield record

        [f.close() for f in f_handles]



class EMGNPYDataset(Dataset):
    def __init__(self, data_path,nfft,hop_length, norm=True,point_detection=False,ss=False,before_mean=False,shuffle=False):
        super(EMGNPYDataset, self).__init__()

        self._data_path = data_path
        self.shuffle = shuffle
        self.nfft = nfft
        self.hop_length = hop_length
        self.normalize = norm
        self.point_detection = point_detection
        self.ss = ss
        self.before_mean = before_mean
        

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)
        
    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line
    
        :type line: str
        """
        line = line.strip()
        
        mfsc_data = np.load(line)

        return Record(mfsc_data, index=mfsc_data.shape[0])


class EMGNPYVocieDataset(Dataset):
    def __init__(self, data_path,nfft,hop_length, norm=True,point_detection=False,ss=False,before_mean=False, voice=1.0, shuffle=False):
        super(EMGNPYVocieDataset, self).__init__()

        self._data_path = data_path
        self.shuffle = shuffle
        self.nfft = nfft
        self.hop_length = hop_length
        self.normalize = norm
        self.point_detection = point_detection
        self.ss = ss
        self.before_mean = before_mean
        self.voice = voice
        

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)
        
    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line
    
        :type line: str
        """
        line = line.strip()
        if re.search('voice', line) is not None:
            weight = self.voice
        else:
            weight = 1.0
        
        mfsc_data = np.load(line)

        return Record((weight, mfsc_data), index=mfsc_data.shape[0])


class EMGDataset(Dataset):
    def __init__(self, data_path,nfft,hop_length, norm=True,point_detection=False,ss=False,before_mean=False,shuffle=False):
        super(EMGDataset, self).__init__()

        self._data_path = data_path
        self.shuffle = shuffle
        self.nfft = nfft
        self.hop_length = hop_length
        self.normalize = norm
        self.point_detection = point_detection
        self.ss = ss
        self.before_mean = before_mean
        

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)
        
    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line
    
        :type line: str
        """
        line = line.strip()
        # data = sio.loadmat(line)
        # raw_data = np.expand_dims(data["data"], axis=0)
        file_name = os.path.split(line)[1]
        label = int(file_name.split(".")[0])
        array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ce.npy"
        if self.before_mean: 
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ce.npy"

        if self.ss:
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ss"+".npy"
            if self.before_mean:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ss"+".npy"

            if self.point_detection:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_pd"+".npy"
                if self.before_mean:
                    array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_pd"+".npy"

        if os.path.exists(array_file):
            mfsc_data = np.load(array_file)
        else:
        #feature = compute_torch_fbank(line)
            data = sio.loadmat(line)
            raw_data = np.expand_dims(data["data"], axis=0)
            filter_x = ceshi2(raw_data, 1000)
            filter_x = np.squeeze(filter_x,axis=0)
            
            #!emg_asr 
            filter_data = filter_x[:,:]
            if self.ss:
                filter_data = SSfilter(filter_x)
                
                if self.point_detection:
                
                    filter_data = SAD(filter_data)

            n_mels = 36
            sr = 1000
            channel_list = []
            channel_num = filter_data.shape[-1]


            for j in range(channel_num):                             # 通道数

                norm_x = filter_data[:, j] 
                tmp = librosa.feature.melspectrogram(norm_x, sr, n_mels=n_mels, n_fft=self.nfft, hop_length=self.hop_length)#! hop_length曾用22、50

                tmp = librosa.power_to_db(tmp).T
                
                channel_list.append(tmp)
            mfsc_data = np.concatenate(channel_list, axis=-1)
            if self.before_mean:

                mfsc_data -= (np.mean(mfsc_data, axis=0) + 1e-8)
            
            if self.normalize:
                mean = np.mean(mfsc_data)
                std = np.std(mfsc_data)
                mfsc_data = (mfsc_data - mean)/std  
            
            np.save(array_file,mfsc_data)
        return Record(mfsc_data, index=mfsc_data.shape[0])


class EMGAddVoiceDataset(Dataset):
    def __init__(self, data_path,nfft,hop_length, norm=True,point_detection=False,ss=False,before_mean=False,voice=1.0,shuffle=False):
        super(EMGAddVoiceDataset, self).__init__()

        self._data_path = data_path
        self.shuffle = shuffle
        self.nfft = nfft
        self.hop_length = hop_length
        self.normalize = norm
        self.point_detection = point_detection
        self.ss = ss
        self.before_mean = before_mean
        self.voice = voice
        

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)
        
    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line
    
        :type line: str
        """
        line = line.strip()

        if re.search('voice', line) is not None:
            weight = self.voice
        else:
            weight = 1.0
        file_name = os.path.split(line)[1]
        label = int(file_name.split(".")[0])
        array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ce.npy"
        if self.before_mean: 
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ce.npy"

        if self.ss:
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ss"+".npy"
            if self.before_mean:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ss"+".npy"

            if self.point_detection:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_pd"+".npy"
                if self.before_mean:
                    array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_pd"+".npy"

        if os.path.exists(array_file):
            mfsc_data = np.load(array_file)
        else:
            data = sio.loadmat(line)
            raw_data = np.expand_dims(data["data"], axis=0)
            filter_x = ceshi2(raw_data, 1000)
            filter_x = np.squeeze(filter_x,axis=0)
            
            #!emg_asr 
            filter_data = filter_x[:,:]
            if self.ss:
                filter_data = SSfilter(filter_x)
                
                if self.point_detection:
                
                    filter_data = SAD(filter_data)
            
            n_mels = 36
            sr = 1000
            channel_list = []
            channel_num = filter_data.shape[-1]


            #!
            for j in range(channel_num):                             # 通道数
                
                norm_x = filter_data[:, j] 
                tmp = librosa.feature.melspectrogram(norm_x, sr, n_mels=n_mels, n_fft=self.nfft, hop_length=self.hop_length)#! hop_length曾用22、50

                tmp = librosa.power_to_db(tmp).T
                channel_list.append(tmp)
            mfsc_data = np.concatenate(channel_list, axis=-1)
            if self.before_mean:

                mfsc_data -= (np.mean(mfsc_data, axis=0) + 1e-8)
            
            if self.normalize:
                mean = np.mean(mfsc_data)
                std = np.std(mfsc_data)
                mfsc_data = (mfsc_data - mean)/std  
            
            np.save(array_file,mfsc_data)
        return Record((weight, mfsc_data), index=mfsc_data.shape[0])



class MoveNodeEMGDataset(Dataset):
    def __init__(self, data_path,nfft,hop_length, norm=True,point_detection=False,ss=False,before_mean=False,shuffle=False,move_channel=[1]):
        super(MoveNodeEMGDataset, self).__init__()

        self._data_path = data_path
        self.shuffle = shuffle
        self.nfft = nfft
        self.hop_length = hop_length
        self.normalize = norm
        self.point_detection = point_detection
        self.ss = ss
        # self.remove_node = move_channel - 1
        self.move_channel = move_channel
        # move_num = len(move_channel)
        move_str = ''
        for item in move_channel:
            move_str += str(item)  
        self.move_str = move_str

        self.before_mean = before_mean
        

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)
        
    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line
    
        :type line: str
        """
        line = line.strip()

        file_name = os.path.split(line)[1]
        label = int(file_name.split(".")[0])
        
        array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ce"+"_mv"+str(self.move_str)+".npy"
        if self.before_mean: 
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ce"+"_mv"+str(self.move_str)+".npy"

        if self.ss:
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ss"+"_mv"+str(self.move_str)+".npy"
            if self.before_mean:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ss"+"_mv"+str(self.move_str)+".npy"

            if self.point_detection:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_pd"+"_mv"+str(self.move_str)+".npy"
                if self.before_mean:
                    array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_pd"+"_mv"+str(self.move_str)+".npy"

        if os.path.exists(array_file):
            mfsc_data = np.load(array_file)
        else:
        #feature = compute_torch_fbank(line)
            data = sio.loadmat(line)
            raw_data = np.expand_dims(data["data"], axis=0)
            filter_x = ceshi2(raw_data, 1000)
            filter_x = np.squeeze(filter_x,axis=0)
            
            #!emg_asr 
            filter_data = filter_x[:,:]
            if self.ss:
                filter_data = SSfilter(filter_x)
                
                if self.point_detection:
                
                    filter_data = SAD(filter_data)
            
            n_mels = 36
            sr = 1000
            channel_list = []
            channel_num = filter_data.shape[-1]

            #!
            for j in range(channel_num): 
                if (j+1) in self.move_channel:
                    continue  
                else:                         # 通道数

                    norm_x = filter_data[:, j] 
                    tmp = librosa.feature.melspectrogram(norm_x, sr, n_mels=n_mels, n_fft=self.nfft, hop_length=self.hop_length)#! hop_length曾用22、50

                    tmp = librosa.power_to_db(tmp).T
                    channel_list.append(tmp)
            mfsc_data = np.concatenate(channel_list, axis=-1)
            if self.before_mean:

                mfsc_data -= (np.mean(mfsc_data, axis=0) + 1e-8)
            
            if self.normalize:
                mean = np.mean(mfsc_data)
                std = np.std(mfsc_data)
                mfsc_data = (mfsc_data - mean)/std  
            
            np.save(array_file,mfsc_data)
        return Record(mfsc_data, index=mfsc_data.shape[0])

class VoiceMoveNodeEMGDataset(Dataset):
    def __init__(self, data_path,nfft,hop_length, norm=True,point_detection=False,ss=False,before_mean=False, voice=1.0, shuffle=False,move_channel=[1]):
        super(VoiceMoveNodeEMGDataset, self).__init__()

        self._data_path = data_path
        self.shuffle = shuffle
        self.nfft = nfft
        self.hop_length = hop_length
        self.normalize = norm
        self.point_detection = point_detection
        self.ss = ss
        self.voice =voice
        self.move_channel = move_channel
        move_str = ''
        for item in move_channel:
            move_str += str(item)  
        self.move_str = move_str

        self.before_mean = before_mean
        

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)
        
    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line
    
        :type line: str
        """
        line = line.strip()
        # data = sio.loadmat(line)
        # raw_data = np.expand_dims(data["data"], axis=0)
        file_name = os.path.split(line)[1]
        label = int(file_name.split(".")[0])

        if re.search('voice', line) is not None:
            weight = self.voice
        else:
            weight = 1.0
        
        array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ce"+"_mv"+str(self.move_str)+".npy"
        if self.before_mean: 
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ce"+"_mv"+str(self.move_str)+".npy"

        if self.ss:
            array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_ss"+"_mv"+str(self.move_str)+".npy"
            if self.before_mean:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_ss"+"_mv"+str(self.move_str)+".npy"

            if self.point_detection:
                array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_pd"+"_mv"+str(self.move_str)+".npy"
                if self.before_mean:
                    array_file = os.path.split(line)[0]+"/"+str(label)+"_"+str(self.nfft)+"_"+str(self.hop_length)+"_before_pd"+"_mv"+str(self.move_str)+".npy"

        if os.path.exists(array_file):
            mfsc_data = np.load(array_file)
        else:
            data = sio.loadmat(line)
            raw_data = np.expand_dims(data["data"], axis=0)
            filter_x = ceshi2(raw_data, 1000)
            filter_x = np.squeeze(filter_x,axis=0)
            
            #!emg_asr 
            filter_data = filter_x[:,:]
            if self.ss:
                filter_data = SSfilter(filter_x)
                
                if self.point_detection:
                
                    filter_data = SAD(filter_data)
            
            n_mels = 36
            sr = 1000
            channel_list = []
            channel_num = filter_data.shape[-1]

            #!
            for j in range(channel_num): 
                if (j+1) in self.move_channel:
                    continue  
                else:                         # 通道数
                
                    norm_x = filter_data[:, j] 
                    tmp = librosa.feature.melspectrogram(norm_x, sr, n_mels=n_mels, n_fft=self.nfft, hop_length=self.hop_length)#! hop_length曾用22、50

                    tmp = librosa.power_to_db(tmp).T
                    channel_list.append(tmp)
            mfsc_data = np.concatenate(channel_list, axis=-1)
            if self.before_mean:

                mfsc_data -= (np.mean(mfsc_data, axis=0) + 1e-8)
            
            if self.normalize:
                mean = np.mean(mfsc_data)
                std = np.std(mfsc_data)
                mfsc_data = (mfsc_data - mean)/std  
            
            np.save(array_file,mfsc_data)
        return Record((weight,mfsc_data), index=mfsc_data.shape[0])



class TextLineDataset(Dataset):
    """
    ```TextDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path,
                 max_len=-1,
                 shuffle=False
                 ):
        super(TextLineDataset, self).__init__()

        self._data_path = data_path
        self._max_len = max_len
        self.shuffle = shuffle

        with open(self._data_path) as f:
            self.num_lines = sum(1 for _ in f)

    @property
    def data_path(self):
        return self._data_path

    def __len__(self):
        return self.num_lines

    def _apply(self, line):
        """
        Process one line

        :type line: str
        """
        line = line.strip().split()
        if 0 < self._max_len < len(line):
            return None

        return Record(line, index=len(line))


class ZipDataset(Dataset):
    """
    ```ZipDataset``` is a kind of dataset which is the combination of several datasets. The same line of all
    the datasets consist on sample. This is very useful to build dataset such as parallel corpus in machine
    translation.
    """

    def __init__(self, *datasets, shuffle=False):
        """
        """
        super(ZipDataset, self).__init__()
        self.shuffle = shuffle
        self.datasets = datasets

    @property
    def data_path(self):
        return [ds.data_path for ds in self.datasets]

    def __len__(self):
        return len(self.datasets[0])

    def _data_iter(self, shuffle):

        if shuffle:
            return shuffle(*self.data_path)
        else:
            return [open(dp) for dp in self.data_path]

    def _apply(self, *lines: str) -> Union[Record, None]:
        """
        :type dataset: TextDataset
        """

        records = [d._apply(l) for d, l in zip(self.datasets, lines)]

        if any([r is None for r in records]):
            return None
        else:
            return zip_records(*records)

