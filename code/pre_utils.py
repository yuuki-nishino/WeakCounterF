from numpy import abs, max 
import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from fastdtw import fastdtw
import seaborn as sns
import math


#自己相関
def autocorrelation(data,k):
    """
    data:自己相関を求めたい時系列データ（1次元データ）
    k:相関を求めるずれの大きさ（＝kだけずらした時系列データとの相関を計算）
    """ 
    
    y_avg = np.mean(data) #データの平均値
    
    #===========ずらしたデータと元のデータとの共分散（相関係数の分母）===========
    sum_of_covariance = 0
    for i in range(k+1,len(data)):
        covariance = (data[i] - y_avg)*(data[i-(k+1)]-y_avg)
        sum_of_covariance += covariance
        
    #===========元のデータの分散^2（ずらしたデータの分散と等しいから）===========
    sum_of_denominator = 0
    for u in range(len(data)):
        denominator = (data[u] - y_avg)**2
        sum_of_denominator += denominator
        
    return sum_of_covariance / sum_of_denominator

#フーリエ変換からのローパスフィルタ
def lowpass(sampling_rate,data,cut_off):
    """
    sampling_rate:サンプリングレート
    data:ローパスフィルタをかける時系列データ（1次元）
    cut_off:カットオフ周波数
    """
    
    sampling_cycle = 1.0/sampling_rate #サンプリング周期
    N = len(data) #サンプル数
    #============== FFT(高速フーリエ変換) ===================
    fft_ampl = np.fft.fft(data)
    fft_time = np.linspace(0,sampling_rate,N)
    #=============== ローパスフィルタ =======================
    cut_off2 = fft_time[-1] - cut_off
    fft_ampl[((fft_time>cut_off)&(fft_time<cut_off2))] = 0 + 0j
    #=============== 逆フーリエ変換 =========================
    ampl = np.fft.ifft(fft_ampl)
    ampl_real = ampl.real
    
    return ampl_real

#フーリエ変換のプロット
def plot_fft(sampling_cycle,ampl):
    """
    sampling_cycle:サンプリング周期
    ampl:時系列データ（1次元）
    """
    
    N = len(ampl) #データ数
    #============FFTを実施し，絶対値の値も取得============
    fft_ampl = np.fft.fft(ampl)
    abs_fft_amp = np.abs(fft_ampl)
    abs_fft_amp = abs_fft_amp / N*2
    abs_fft_amp[0] = abs_fft_amp[0] /2
    
    #==========横軸（周波数）の配列とピーク部分の取得==========
    frequency = np.linspace(0,1.0/sampling_cycle,N)
    maximal_idx = signal.argrelmax(abs_fft_amp[:int(N/2)+1],order=1)
    
    #==================プロット==================
    plt.figure(figsize=(10,8))
    plt.plot(frequency[:int(N/2)+1],abs_fft_amp[:int(N/2)+1])
    plt.scatter(frequency[maximal_idx],abs_fft_amp[maximal_idx],c='red',s=25)
    plt.grid(True)
    plt.title('Fast Fourier Transform')
    plt.xlabel('frequency[Hz]')
    plt.ylabel('amplitude')
    
    return frequency,fft_ampl

#ローパスフィルタのプロット

def plot_low_pass_filter(fft_time,fft_amp,cut_off):
    """
    fft_time:周波数を示す配列
    fft_amp:各周波数におけるスペクトルを示す配列
    cut_off:ローパスフィルタを行う際のカットオフ周波数
    """
    
    N = len(fft_amp) #データ数
    cut_off2 = fft_time[-1] - cut_off #最大周波数からカットオフ周波数を減算（カットオフ周波数2）
    
    #=============カットオフ周波数以上かつカットオフ周波数2以下の値を0に設定=============
    fft_amp[((fft_time>cut_off)&(fft_time<cut_off2))] = 0 + 0j
    abs_fft_amp = np.abs(fft_amp)
    abs_fft_amp = abs_fft_amp / N*2
    abs_fft_amp[0] = abs_fft_amp[0] / 2
    
    maximal_idx = signal.argrelmax(abs_fft_amp,order=1) #ピーク部分を取得
    
    #==========================プロット==========================
    plt.figure(figsize=(10, 8))
    plt.plot(fft_time, abs_fft_amp)
    plt.scatter(fft_time[maximal_idx], abs_fft_amp[maximal_idx], c='red', s=25)
    plt.grid(True)
    plt.title('Low Pass Filter')
    plt.xlabel('freqency[Hz]')
    plt.ylabel('amplitude')
    
    return fft_time,fft_amp

#逆フーリエ変換のプロット
def plol_ifft(fft_time,fft_ampl):
    """
    fft_time:周波数を示す配列
    fft_ampl:ローパスフィルタ後のスペクトル配列
    """
    N = len(fft_ampl) #データ数
    
    #==============逆フーリエ変換==============
    ampl = np.fft.ifft(fft_ampl)
    ampl_real = ampl.real
    
    #==============時系列を示す配列（横軸用）==============
    frequency = fft_time[-1]
    time = np.arange(0,1.0/frequency*N,1.0/frequency)
    
    #==================プロット==================
    plt.figure(figsize=(20,5))
    plt.plot(ampl_real)
    plt.grid(True)
    plt.title('Invese Fast Fourier Transform')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    
    return time,ampl_real


#移動平均
def moving_average(data,window_size):
    """
    data:平滑化する時系列データ（1次元）
    window_size:移動平均をかけるスライディングウィンドウのサイズ
    """
    v = np.ones(window_size)/window_size
    moving_average = np.convolve(data,v,mode='same')
    return moving_average
