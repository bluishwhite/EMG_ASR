from scipy import signal
import scipy.io as sio
import librosa
import numpy as np


# 谱减法滤波
def SpectralSub(signal, wlen, inc, NIS, a, b):
    wnd = np.hamming(wlen)
    y = enframe(signal, wnd, inc)
#     print(y)
#     print(y.shape)
    fn, flen = y.shape
    y_a = np.abs(np.fft.fft(y, axis=1))
    
    y_a2 = np.power(y_a, 2)

    
#     print(y_a2.shape)
    y_angle = np.angle(np.fft.fft(y, axis=1))
    Nt = np.mean(y_a2[:NIS, ], axis=0)
#     print("Nt.shape:", Nt.shape)
    

    y_a2 = np.where(y_a2 >= a * Nt, y_a2 - a * Nt, b * Nt)
    y_a2 = np.sqrt(y_a2)

    X = y_a2 * np.cos(y_angle) + 1j * y_a2 * np.sin(y_angle)
    hatx = np.real(np.fft.ifft(X, axis=1))
#     print(hatx.shape)

    sig = np.zeros(int((fn - 1) * inc + wlen))

    for i in range(fn):
        start = i * inc
        sig[start:start + flen] += hatx[i, :]
    return sig

def SSfilter(filt_x):
    fs = 1000
    sub_filt_x = []
    
    tmp = []
    for j in range(filt_x.shape[1]):
        data = filt_x[:, j]
#             data = data - np.mean(data)
#             data = data / np.max(np.abs(data))

        IS = 0.25  # 设置前导无话段长度
        wlen = 128  # 设置帧长
        inc = 48  # 设置帧移
        N = len(data)  # 信号长度
        time = [i / fs for i in range(N)]  # 设置时间
        NIS = int((IS * fs - wlen) // inc + 1)
#             print(NIS)
        a, b = 4, 0.001

        output = SpectralSub(data, wlen, inc, NIS, a, b)
        if len(output) < len(data):
            filted = np.zeros(len(data))
            filted[:len(output)] = output
        elif len(output) > len(data):
            filted = output[:len(data)]
        else:
            filted = output
#             filted = filted/np.max(abs(filted))
        sub_filt_x.append(filted.reshape(-1,1))
        # sub_filt_x.append(np.concatenate(tmp, axis=-1))
    sub_filt_x = np.concatenate(sub_filt_x, axis=-1)
    return sub_filt_x

def SAD(data):
    semg = data
    # print(semg)
    # print(semg.shape)
    # (2950, 8)
    act_flag = 0
    jg_flag = 0
    window_len = 25 #30

    act_data = []
    data_act_sel = []

    # 先计算阈值，得到基准的基线阈值；
    m = 250 # 250
    semg_baseline = semg[:m, [0,1,2,3,4,5]] # 取了四个通道 # (150,3)

    semg_baseline = semg_baseline.mean(axis=1) # (150,) 同一时刻，三个通道在这一时刻的平均值

    # np.square计算平方
    amp = np.mean(np.sum(np.square(np.square(semg_baseline[0:25])))+np.sum(np.square(semg_baseline[25:50])))
    # print('amp is', amp) # amp is 
    # print(round((semg.shape[0]) / window_len)) # 40
    # Round函数返回一个数值，该数值是按照指定的小数位数进行四舍五入运算的结果
    for i in range(round((semg.shape[0]) / window_len)):   
        temp = np.array(semg[i * window_len:(i + 1) * window_len, [0,1,2,3,4,5]])
        temp = temp.mean(axis=1) 
        temp = np.sum(np.square(temp))
        # print(temp)
        #! A = 0.13
        A = 350
        if temp >= A * amp:   # 阈值设为八
            # print('data is act')
            act_flag = 1
            jg_flag = 0
            # for j in range(64):
            act_data = semg[i * window_len:(i + 1) * window_len, :]
            act_data = act_data.tolist()
            # tolist() 函数 用于将数组或矩阵转换成列表 
            data_act_sel.extend(act_data)

        else:
            # print('data is rest')
            if act_flag == 1:
                if jg_flag < 3:  # 冷却长度设为3个帧
                    act_data = semg[i * window_len:(i + 1) * window_len, :]
                    act_data = act_data.tolist()
                    data_act_sel.extend(act_data)
                    # print(act_data_all.shape)
                    # print(act_data.shape)
                    # np.concatenate((act_data_all, act_data), axis=0)
                    jg_flag += 1
                else:
                    continue
                    # print('get a sample')
                #     if len(act_data[]) > 20 * 8 * 30:  # 样本最小长度限定为20个帧
                #         yangben_flag += 1
                #         act_data.append([])
                #     act_flag = 0
                #     jg_flag = 0

    data_act_sel = np.array(data_act_sel)

    return data_act_sel


# vocab_dict = get_vocab("debug/183_8_1_1data/train_183_data.de")
# train_frequencies = word_count("debug/183_8_1_1data/train_183_data.de",vocab_dict)
# dev_frequencies = word_count("debug/183_8_1_1data/test_183_data.de",vocab_dict)
# print(word_freq_rmse(train_frequencies,dev_frequencies))

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
#     print(indf)
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


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

def display_picture(line):
    """
    Process one line

    :type line: str
    """
    line = line.strip()

    #feature = compute_torch_fbank(line)
    data = sio.loadmat(line)
    raw_data = np.expand_dims(data["data"], axis=0)

    filter_x = ceshi2(raw_data, 1000)
    filter_x = np.squeeze(filter_x,axis=0)

    