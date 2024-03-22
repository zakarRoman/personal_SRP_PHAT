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


def srp_phat(s, L, fs, nfft=None, mode='far'):
    if nfft is None:
        nfft = 1024

    # 实例化STFT，傅立叶变换对象
    stft_instance = pra.transform.stft.STFT(nfft, hop=nfft // 2)
    # 对每个信号执行STFT分析
    s_FFT = np.array([stft_instance.analysis(signal).T for signal in s])

    # s_FFT = np.array([pra.stft(s, nfft, nfft // 2, transform=np.fft.rfft).T for s in s])

    # 创建一个SRP类
    doa = pra.doa.srp.SRP(L, fs, nfft, c=343.0, num_src=1, mode=mode, r=None, azimuth=None, colatitude=None)

    # 进行SRP
    doa.locate_sources(s_FFT)

    # 使用自带的方法画图
    print('SRP-PHAT')
    print('Speakers at: ', np.sort(doa.azimuth_recon) / np.pi * 180, 'degrees')



if __name__ == '__main__':
    # print(f'数据形状为{x_1.shape}')
    # print(f'fs大小为{fs_1}')

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
