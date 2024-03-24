# 导入需要的库
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import librosa


# 写一个函数用来读取数据
def read_audio(filename):
    _fs, _y = wav.read(filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # 短整型化为浮点型
    return _y, _fs


def calculate_frequencies(yWin, fs):  # yWin代表信号在一个窗口时间段内的采样点, fs是信号的采样频率，表示每秒的采样次数
    freq = np.fft.fftfreq(len(yWin), 1.0 / fs)  # 窗口内数据的频率
    freq = freq[1:]
    return freq


def calculate_energy(yWin, fs):
    ampl = np.abs(np.fft.fft(yWin))  # 快速傅立叶变换，abs用于获取结果的幅度，用来表示信号的能量
    ampl = ampl[1:]  # 去掉幅度数组中的直流分量
    energy = ampl ** 2  # 幅度数组平方，获得能量
    return energy


# 声音检测函数，用于处理语音信号
def vad(y, fs, thrs):
    # thrs:阈值
    # 设置帧长度以及重叠
    frame_length = 0.02
    frame_overlap = 0.01
    energy_thresh = thrs

    frame = int(fs * frame_length)
    frame_overlap = int(fs * frame_overlap)

    detected_windows = np.array([])

    si = 0  # 开始处理的索引
    # 定义处理的语音频率范围（人声频率范围内）
    s_band = 300
    e_band = 3400

    # 遍历数据，以帧为单位进行处理
    while si < (len(y) - frame):  # 以帧为单位对信号进行处理
        ei = si + frame  # 窗口结束索引

        if ei >= len(y): ei = len(y) - 1  # 如果超出了数据末尾，就将结束帧设为数据末尾
        yWin = y[si:ei]  # 取走数据帧

        freqs = calculate_frequencies(yWin, fs)  # 计算这些帧的频率
        energy = calculate_energy(yWin, fs)  # 计算这些帧的能量

        # 使用能量 - 频率字典，用于存储每个频率成分的能量
        energy_freq = {}
        for (i, freq) in enumerate(freqs):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = energy[i] * 2

        # 计算最大能量
        sum_max_energy = sum(energy_freq.values())

        # 计算语音带的总能量
        sum_energy = 0
        for f in energy_freq.keys():
            if s_band < f < e_band:  # 频率在人声范围内
                sum_energy += energy_freq[f]

        # 计算比率
        ratio = sum_energy / sum_max_energy
        # 与阈值结合进行判断
        ratio = ratio > energy_thresh
        # 输出每帧是否检测到语音的结果
        detected_windows = np.append(detected_windows, [si, ratio])
        si += frame_overlap  # 移动到下一帧

    # 重塑输出数据
    detected_windows = detected_windows.reshape(len(detected_windows) // 2, 2)
    return detected_windows


# srp-phat函数
def srp_phat(s, L, fs, nfft=None, mode='far'):
    if nfft is None:
        nfft = 1024

    # 实例化STFT，傅立叶变换对象
    stft_instance = pra.transform.stft.STFT(nfft, hop=nfft // 2)
    # 对每个信号执行STFT分析
    s_FFT = np.array([stft_instance.analysis(signal).T for signal in s])

    # 创建一个SRP类
    doa = pra.doa.srp.SRP(L, fs, nfft, c=343.0, num_src=1, mode=mode, r=None, azimuth=None, colatitude=None)

    # 进行SRP
    doa.locate_sources(s_FFT)
    print(f'方位角的上srp的值为:{doa.grid.values}')
    matrix = np.array(doa.grid.values)
    print(matrix.shape)

    # doa.polar_plt_dirac()
    # plt.title('SRP_PHAT')
    # 使用自带的方法画图
    # print('Speakers at: ', np.sort(doa.azimuth_recon) / np.pi * 180, 'degrees')
    # plt.show()


if __name__ == '__main__':
    # 通过文档获得的第一个麦克风阵列坐标
    L_1 = np.array([[-0.1, 0.4, 0],  # 1
                    [-0.07071, 0.32929, 0],  # 2
                    [0, 0.3, 0],  # 3
                    [0.07071, 0.32939, 0],  # 4
                    [0.1, 0.4, 0],  # 5
                    [0.07071, 0.47071, 0],  # 6
                    [0, 0.5, 0],  # 7
                    [-0.07071, 0.47071, 0]]
                   ).T
    print(L_1.shape[1])

    x_1, fs_1 = read_audio('seq11-1p-0100_array1_mic1.wav')
    x_2, fs_2 = read_audio('seq11-1p-0100_array1_mic2.wav')
    x_3, fs_3 = read_audio('seq11-1p-0100_array1_mic3.wav')
    x_4, fs_4 = read_audio('seq11-1p-0100_array1_mic4.wav')
    x_5, fs_5 = read_audio('seq11-1p-0100_array1_mic5.wav')
    x_6, fs_6 = read_audio('seq11-1p-0100_array1_mic6.wav')
    x_7, fs_7 = read_audio('seq11-1p-0100_array1_mic7.wav')
    x_8, fs_8 = read_audio('seq11-1p-0100_array1_mic8.wav')
    X = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]
    X = np.array(X)

    srp_phat(s=X, L=L_1, fs=fs_1)
