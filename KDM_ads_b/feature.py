# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:54:38 2023
"""
import numpy as np
from scipy.stats import zscore
"from Higher_Moment import Higher_Moment"

def Feature_deal(BPSK):
    Feature = np.zeros(26)

    Feature[0] = InsAmp(BPSK) # 门限数量

    # 进行归一化
    StdBPSK = zscore(np.abs(BPSK))
    # StdBPSK = (np.abs(BPSK) - np.mean(np.abs(BPSK))) / np.std(np.abs(BPSK)) 公式处理

    Feature[1] = InAmStd(StdBPSK) # 瞬时幅度的标准差
    Feature[2] = InAmAStd(StdBPSK) # 瞬时幅度的绝对标准差
    Feature[3] = InAmKur(StdBPSK) # 瞬时幅度的峰度

    InsFre = PhaseDeal(BPSK) # 信号的瞬时频率
    Feature[4] = InAmAStd(InsFre) # 瞬时频率的绝对标准差
    Feature[5] = InAmKur(InsFre) # 瞬时频率的峰度
    
    # 提取统计特征
    ##函数输出形式影响以下语句形式
    Feature[6:10] = Higher_Moment(BPSK)
    
    # 提取频谱特征
    Feature[10] = MaxSpectral(StdBPSK) # 频谱最大值
    ## 原代码语句SpeFea6 = MaxSpectral(StdQPSK);

    Feature[11] = SpeSym(BPSK )# 频谱对称性
    Feature[12] = SpeThree(BPSK) # 频谱3dB带宽
    Feature[13] = SpeVar(BPSK) # 频谱的方差
    ## 原代码语句：SpeFea4 = SpeThree(SCF);
    
    # 频谱的三个局部最大值标签
    Feature[14:17] = FindMax(BPSK, 1)
    Feature[17:20] = FindMax(BPSK, 2)
    Feature[20:23] = FindMax(BPSK, 4)
    Feature[23:26] = FindMax(BPSK, 8)

    return Feature

# 瞬时幅度门限数量
def InsAmp(Signal):
    threshold = np.abs((Signal - np.mean(Signal)) / np.std(Signal))

    count = np.sum((threshold > 0.4) & (threshold < 0.8))
    return count / len(Signal)

# 瞬时幅度的标准差
def InAmStd(Signal):
    N = len(Signal)
    return np.sqrt(np.sum(Signal ** 2) / N - (np.sum(Signal) / N) ** 2)

# 瞬时幅度的绝对的标准差
def InAmAStd(Signal):
    N = len(Signal)
    return np.sqrt(np.sum(Signal ** 2) / N - (np.sum(np.abs(Signal)) / N) ** 2)

# 瞬时频率的峰度
def InAmKur(Signal):
    return np.mean(Signal ** 4) / (np.mean(Signal ** 2)) ** 2

# 信号的最大频谱值
def MaxSpectral(BPSK):
    L = len(BPSK)
    NFFT = 2 ** int(np.ceil(np.log2(L)))
    Y = np.fft.fft(BPSK, NFFT) / L
    return np.max(np.abs(Y))

# 寻找频谱对称性
def SpeSym(Signal):
    L = len(Signal)
    y = np.fft.fftshift(np.fft.fft(Signal, L))
    U = L // 2
    y1 = y[:U]
    y2 = y[U:L]
    PL1 = np.sum(np.abs(y1) ** 2)
    PL2 = np.sum(np.abs(y2) ** 2)
    return (PL1 - PL2) / (PL1 + PL2)

# 频谱3dB带宽
def SpeThree(Signal):
    L = len(Signal)
    NFFT = 2 ** int(np.ceil(np.log2(L)))
    Y = np.fft.fft(Signal, NFFT) / L
    ECHO1 = np.abs(Y) / np.max(np.abs(Y))
    ECHO2 = 20 * np.log10(ECHO1)
    return np.sum(ECHO2 >= -3)

# 频谱的方差
def SpeVar(Signal):
    L = len(Signal)
    NFFT = 2 ** int(np.ceil(np.log2(L)))
    Y = np.fft.fft(Signal, NFFT) / L
    return np.std(Y)

# 寻找局部最大值
def FindMax(BPSK, p):
    # BPSK代表信号，P代表信号的几次幂
    L = len(BPSK)
    NFFT = 2 ** int(np.ceil(np.log2(L)))
    y = np.abs(np.fft.fft(np.power(BPSK, p), NFFT))
    peak, index = [], []

    for i in range(L - 2):
        if y[i + 1] > y[i] and y[i + 1] > y[i + 2]:
            peak.append(y[i + 1])
            index.append(i + 1)

    MaxIndex = np.argsort(peak)[::-1]
    f1 = index[MaxIndex[0]] / L
    f2 = index[MaxIndex[1]] / L
    f3 = index[MaxIndex[2]] / L

    return f1, f2, f3

# 根据信号的实据虚部转换为相应的相位
def PhaseDeal(BPSK):
    N = len(BPSK)
    tempphase = np.zeros(N)
    Fre = np.zeros(N)
    
    # for i in range(N):
    #     tempphase[i] = np.angle(BPSK[i])
    for i in range(N):
        if np.imag(BPSK[i]) == 0 and np.real(BPSK[i])>0:
            tempphase[i] = 0
        elif np.imag(BPSK[i]) > 0 and np.real(BPSK[i])>0:
            tempphase[i] = np.angle(BPSK[i])
        elif np.imag(BPSK[i]) > 0 and np.real(BPSK[i])<0:
            tempphase[i] = np.pi/2 + np.angle(BPSK[i])
        elif np.imag(BPSK[i]) > 0 and np.real(BPSK[i])==0:
            tempphase[i] = np.pi/2
        elif np.imag(BPSK[i]) == 0 and np.real(BPSK[i])<0:
            tempphase[i] = np.pi
        elif np.imag(BPSK[i]) < 0 and np.real(BPSK[i])<0:
            tempphase[i] = np.pi + np.angle(BPSK[i])
        elif np.imag(BPSK[i]) < 0 and np.real(BPSK[i])==0:
            tempphase[i] = 3*np.pi/2 
        else:
            tempphase[i] = 3*np.pi/2 - np.angle(BPSK[i])
 
    phase = np.concatenate(([0], tempphase))
    Fre = np.diff(phase)
    InsFre = Fre - np.mean(Fre)

    return InsFre

# 原代码函数angle（h）、Random（min，max）可省略
def Higher_Moment(signal):
    M20 = gaojieju(signal, 2, 0)
    M21 = gaojieju(signal, 2, 1)
    M40 = gaojieju(signal, 4, 0)
    M42 = gaojieju(signal, 4, 2)
    M60 = gaojieju(signal, 6, 0)
    M63 = gaojieju(signal, 6, 3)
    M80 = gaojieju(signal, 8, 0)
    [C21, C40, C42, C60, C63, C80] = gaojieleijiliang(M20, M21, M40, M42, M60, M63, M80)
    C21_abs = np.abs(C21)
    C40_abs = np.abs(C40)
    C42_abs = np.abs(C42)
    C60_abs = np.abs(C60)
    C63_abs = np.abs(C63)
    C80_abs = np.abs(C80)
    """    % F1 = (C42_abs + C40_abs) / (C21_abs ^ 2);
    % F2 = (C42_abs - C40_abs) / (C21_abs ^ 2);
    % F3 = (C63_abs ^ 4 + C80_abs ^ 3) / (C21_abs ^ 12);
    % F4 = C63_abs ^ 2 / C40_abs ^ 3;"""
    F1 = C42_abs
    F2 = C40_abs / C42_abs
    F3 = (C60_abs ** 2) / (C42_abs ** 3)
    F4 = C80_abs / C42_abs ** 2
    return F1,F2,F3,F4

def gaojieju(signal,p,q):
    signal_conj = signal.conjugate()
    return np.mean(signal ** (p - q) * signal_conj ** q)

def gaojieleijiliang(M20, M21, M40, M42, M60, M63, M80):
    C21 = M21
    C40 = M40 - 3 * M20 ** 2
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    C60 = M60 - 15* M40 * M20 + 30. * M20 ** 3
    C63 = M63 - 9 * C42 * C21 - 6 * C21 ** 2
    C80 = M80 - 28. * M60 * M20 - 35 * M40 ** 3 + 420 * M40 * M20 ** 2 - 630 * M20 ** 4
    return C21,C40,C42,C60,C63,C80
