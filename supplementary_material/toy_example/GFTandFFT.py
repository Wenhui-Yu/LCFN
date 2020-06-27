import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def GFT(x):
    ## graph Fourior transform
    n = np.size(x)
    I = np.identity(n)
    D = 2 * np.ones(n)
    D[0] = 1
    D[-1] = 1
    A = np.zeros((n,n))
    for i in range(n-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    L = I - np.dot(np.dot(np.diag(D**-0.5), A), np.diag(D**-0.5)) ## Laplacian matrix
    # print(L)
    lamda, P = np.linalg.eigh(L)
    x = np.dot(x, P)
    return lamda, x

## signals in the time domain
x1=np.array(range(20))
# y = np.sin(1*2*np.pi/20*x1) ## f = 1
y = np.sin(7*2*np.pi/20*x1)  ## f = 7
# y = np.sin(2*np.pi/20*x1) + np.sin(7*2*np.pi/20*x1)  ## mixed

## transform
ft=np.abs(fft(y)) # Fast Fourior tranform
x2, gft=np.abs(GFT(y)) # graph Fourior transform
# print(gft)
# for i in x1:
#     print(i)

## plot and save
path = ''
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
plt.subplot(221)
plt.stem(x1,y,linefmt='black',markerfmt = 'ko', basefmt='grey', use_line_collection=True)
plt.xlim([-0.5,20.5])
plt.ylim([-1.8,1.8])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r'Time index $n$', fontsize=17)
plt.ylabel(r'Amplitude', fontsize=17)

plt.subplot(222)
figsize(12.5, 4)
plt.stem(x1,abs(ft),linefmt='black',markerfmt = 'ko', basefmt='grey', use_line_collection=True)
plt.xlim([-0.5,10.5])
plt.ylim([0,11])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r'Frequency index $k$', fontsize=17)
plt.ylabel(r'Amplitude', fontsize=17)

plt.subplot(223)
plt.stem(x2,gft,linefmt='black',markerfmt = 'ko', basefmt='grey', use_line_collection=True)
plt.xlim([-0.1,2.1])
plt.ylim([0,3])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r'Frequency $\lambda_k$', fontsize=17)
plt.ylabel(r'Amplitude', fontsize=17)

plt.savefig(path + 'FFT7.pdf')
plt.show()



