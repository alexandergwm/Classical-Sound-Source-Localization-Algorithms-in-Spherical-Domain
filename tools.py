import numpy as np
import soundfile as sf
from scipy import signal
import os
import scipy.special as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.special import factorial, lpmv
from plot_tools import plot_SSL_results
import pyroomacoustics as pra

def soundread(sound_filepath):
    """
    Returns the contents of a sound file
    :param sound_filepath: path to sound_file to be read
    :return: (signal, sampling rate, number of channels)
    """
    mic_signals, fs = sf.read(sound_filepath, dtype='float32')
    if len(mic_signals.shape) == 1:
        num_channels = 1
    else:
        num_channels = mic_signals.shape[0]
    return mic_signals, fs, num_channels

def sph2cart(r, theta, phi):
    """
    Converts coordinate in spherical coordinates to Cartesian coordinates
    :param r: Radius
    :param theta: Azimuth angle
    :param phi: Inclination angle
    :return: Coordinates in Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    car_coor = np.array([x, y, z])
    return car_coor


def cart2sph(coords):
    """
    Function to convert cartesian coordinates to spherical coordinates.
    :param coords: A np.ndarray of cartesian coordinates
    :return: A np.ndarray of spherical coordinates (r, theta, phi)
    """
    # If input is a single point, add an extra dimension for consistency
    if coords.ndim == 1:
        coords = coords[None, :]
    # Separate x, y, and z for readability
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # Compute radial distance (r)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Compute elevation angle (theta), range is 0 to pi
    theta = np.arccos(z / r)
    # Compute azimuth angle (phi), range is 0 to 2pi
    phi = np.arctan2(y, x) % (2 * np.pi)
    # Combine r, theta, and phi into a single array and return
    return np.stack((r, theta, phi), axis=-1)


def unitSph2cart(aziElev):
    """
    根据方位角和高度角获取单位向量的笛卡尔坐标。
    与sph2cart相似，但假设单位向量，并使用矩阵进行输入和输出，
    而不是分开的坐标向量。

    参数:
    aziElev (numpy.ndarray): 方位角-高度角对。形状应为 (N, 2)。

    返回值:
    numpy.ndarray: 形状为 (N, 3) 的笛卡尔坐标
    """
    if aziElev.shape[1] != 2:
        aziElev = aziElev.T

    azimuth = aziElev[:, 0]
    elevation = aziElev[:, 1]

    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)

    xyz = np.column_stack((x, y, z))

    return xyz


# define the signal
def generateSin():
    duration = 5
    sample_rate = 16000
    t = np.linspace(0,duration,duration * sample_rate)
    f = 500
    sine_wave = np.sin(2 * np.pi * f * t)
    signal = sine_wave
    return signal

# Coordinate correction
def coordinateCorrection(room_dim, mic_pos_car):
    mic_array_center = room_dim/2
    mic_array_center = mic_array_center.reshape(1, -1)
    mic_pos_car = mic_pos_car+mic_array_center
    return mic_pos_car

# Add required SNR noise
def addNoise(signal, snr):
    # Convert SNR from dB scale to linear scale
    snr_linear = 10**(snr/10)
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)
    # Calculate noise power based on signal power and SNR
    noise_power = signal_power / snr_linear
    # Generate noise with calculated power
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, signal.shape)
    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal


def detect_signal_type(signal,fs, threshold_ratio=0.5):
    """
    This function is used to determine if the signal is single frequency or multiple
    :param signal: input signal
    :param fs: sampling rate
    :param threshold_ratio:  The ratio between the maximum magnitude and other magnitudes in frequency domain
    :return: "Single" and its frequency or "Multiple"
    """
    window = np.hanning(len(signal))
    A = np.fft.fft(signal)
    #
    A = A[:len(A)//2]

    # 计算频率成分的绝对值
    magnitude = np.abs(A)

    # 找出最大频率分量的索引
    max_index = np.argmax(magnitude)

    # 计算阈值（最大值的一部分）
    threshold = threshold_ratio * magnitude[max_index]

    # 移除最大频率成分
    magnitude[magnitude == magnitude[max_index]] = 0

    # 找出剩余部分是否有超过阈值的频率分量
    second_peak = np.max(magnitude)

    # 如果剩余部分没有超过阈值的频率分量，则判断为单频信号，否则为多频信号
    if second_peak < threshold:
        frequency = max_index * fs / (len(A)*2)  # convert the index to actual frequency
        return "Single", frequency
    else:
        return "Multiple", None



def aziElev2aziPolar(dirs):
    """
    Convert from azimuth-inclination to azimuth-elevation
    :param dirs: np.ndarray, shape (N, 2), each row is [azimuth, inclination]
    :return: np.ndarray, shape (N, 2), each row is [azimuth, elevation]
    """
    return np.array([np.pi-dirs[:, 0], np.pi/2 - dirs[:, 1]]).T

def grid2dirs(aziRes, polarRes, POLAR_OR_ELEV=True, ZEROED_OR_CENTERED=True):
    """
    Create a vector of a spherical grid directions based on a given azimuthal
    and polar resolution.
    """
    if 360 % aziRes != 0 or 180 % polarRes != 0:
        raise ValueError('Azimuth or elevation resolution should divide exactly 360 and 180 degree')

    if ZEROED_OR_CENTERED:
        phi = np.linspace(0, 2*np.pi, int(360/aziRes), endpoint=False)
    else:
        phi = np.linspace(-np.pi, np.pi, int(360/aziRes), endpoint=False)

    if POLAR_OR_ELEV:
        theta = np.linspace(0, np.pi, int(180/polarRes), endpoint=False)
    else:
        theta = np.linspace(-np.pi/2, np.pi/2, int(180/polarRes) + 1)

    Nphi = len(phi)
    Ntheta = len(theta)

    dirs = []
    for i in range(Ntheta):
        for j in range(Nphi):
            dirs.append([phi[j], theta[i]*np.ones_like(phi)[j]])


    # if POLAR_OR_ELEV:
        # dirs.insert(0, [0, 0])
        # dirs.append([0, np.pi])

    #else:
        #dirs.insert(0, [0, -np.pi / 2])
        #dirs.append([0, np.pi / 2])

    return np.array(dirs)


def SphHarmonic(n, theta, phi):
    y = np.zeros(2*n+1, dtype=complex)
    for m in range(-n, n+1):
        temp = np.sqrt((2*n+1)/(4*np.pi) * factorial(n-abs(m)) / factorial(n+abs(m))) * np.exp(1j*m*phi)
        if m >= 0:
            y[m+n] = temp * lpmv(m, n, np.cos(theta))
        else:
            fm = -m
            y[m+n] = temp * lpmv(fm, n, np.cos(theta)) * (-1)**fm
    return y

def ssl_SHmethod(MicrophoneArray, Theta_l, Phi_l, method, sphere_config, plot_method):
    """
    This script is stored some classical SSL algorithms for source localization
    :param MicrophoneArray:  The Microphone array properties
    :param Theta_l:  The elevation of the source
    :param Phi_l:  The azimuth of the source
    :param method:  The chosen algorithm
    :param sphere_config: The configuration of sphere
    :param plot_method: The method of plotting (2D or 3D)
    :return:  A figure
    """
    radius = MicrophoneArray._radius
    c = 343       # Velocity of sound
    num_mic = MicrophoneArray._numelements
    ka = 4.496   # k = 2* np.pi * f / c      # if f > 2607, N_order = 4
    N = 4
    Mic_Theta = MicrophoneArray._thetas
    Mic_R = MicrophoneArray._radius
    Mic_Phi = MicrophoneArray._phis

    # Calculate bn(ka)
    if sphere_config == "rigid":
        bn = np.zeros(51, dtype=complex)
        for n in range(51):
            jn_ka = sp.spherical_jn(n, ka)
            jn_ka_der = sp.spherical_jn(n, ka, derivative=True)
            yn_ka = sp.spherical_yn(n, ka)
            yn_ka_der = sp.spherical_yn(n, ka, derivative=True)

            # Compute the second kind of Hankel function and its derivative
            hn2_ka = jn_ka - 1j * yn_ka
            hn2_ka_der = jn_ka_der - 1j * yn_ka_der

            bn[n] = 4 * np.pi * (1j) ** n * (jn_ka - jn_ka_der / hn2_ka_der * hn2_ka)

    if sphere_config == "open":
        bn = np.zeros(51, dtype=complex)
        for n in range(51):
            bn[n] = 4 * np.pi * (1j) ** n * sp.spherical_jn(n, ka)
    # bn = calc_bn(ka, 51)
    # Calculate microphone pressures for a single frequency
    p = np.zeros((num_mic, 1), dtype=complex)
    for m in range(num_mic):
        for n in range(51):
            Bn = np.eye(2 * n + 1, dtype=complex) * bn[n]
            p[m] += SphHarmonic(n, Theta_l, Phi_l).conj().T @ Bn @ SphHarmonic(n, Mic_Theta[m], Mic_Phi[m])

    # Add noise
    p = addNoise(p, 20)

    # Calculate spherical harmonics
    Y_nm = np.zeros(((N + 1) ** 2, num_mic), dtype=complex)
    for num in range(num_mic):
        for n in range(N + 1):
            Y_nm[n ** 2:(n + 1) ** 2, num] = SphHarmonic(n, Mic_Theta[num], Mic_Phi[num])

    # Transform signal from frequency domain to spherical harmonic domain
    p_nm = 4 * np.pi / num_mic * Y_nm.conj() @ p

    theta = np.arange(0, np.pi, 1 / 180 * np.pi)
    phi = np.arange(0, 2 * np.pi, 1 / 180 * np.pi)
    if method == "PWD":
        Out = np.zeros((len(theta), len(phi)), dtype=complex)
        B_n = np.zeros(((N + 1) ** 2), dtype=complex)
        D_nm = np.zeros(((N + 1) ** 2, 1), dtype=complex)
        for num1 in range(len(theta)):
            for num2 in range(len(phi)):
                for n in range(N + 1):
                    B_n[n ** 2:(n + 1) ** 2] = 1 / bn[n]
                    D_nm[n ** 2:(n + 1) ** 2, 0] = SphHarmonic(n, theta[num1], phi[num2])
                Out[num1, num2] = D_nm.T @ np.diag(B_n) @ p_nm

        # Convert to dB and clip values below -22 dB
        out = 20 * np.log10(np.abs(Out))
        out = out - np.max(out)
        x, y = np.where(out == np.max(out))
        out = np.clip(out, -22, None)
        source_pos = np.array([x, y])

    if method == 'SHMVDR':
        resolution = 1
        sphCOV = np.dot(p_nm, p_nm.T.conj())
        lambda_reg = 1e-2  # 正则化因子，需要根据你的应用进行调整
        sphCOV += lambda_reg * np.eye(sphCOV.shape[0])
        # print(sphCOV.shape)
        out = []
        P_mvdr = np.zeros((len(theta), len(phi)), dtype=complex)  # MUSIC谱
        for num1 in range(len(theta)):
            for num2 in range(len(phi)):
                U_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                for n in range(N + 1):
                    U_nm[n ** 2:(n + 1) ** 2] = bn[n] * SphHarmonic(n, theta[num1], phi[num2]).conj()
                U_nm = U_nm[:, np.newaxis]
                invA_b = np.linalg.solve(sphCOV, U_nm)
                b_invA_b = np.dot(U_nm.T.conj(), invA_b)
                w_mvdr = invA_b / b_invA_b
                P_mvdr[num1, num2] = w_mvdr.T.conj() @ sphCOV @ w_mvdr

        # Convert to dB and clip values below -22 dB
        out = 10 * np.log10(np.abs(P_mvdr))
        out = out - np.max(out)
        x, y = np.where(out == np.max(out))
        source_pos = np.array([x,y])

    if method == "DAS":
        Out = np.zeros((len(theta), len(phi)), dtype=complex)
        B_n = np.zeros(((N + 1) ** 2), dtype=complex)
        D_nm = np.zeros(((N + 1) ** 2, 1), dtype=complex)
        for num1 in range(len(theta)):
            for num2 in range(len(phi)):
                for n in range(N + 1):
                    D_nm[n ** 2:(n + 1) ** 2, 0] = bn[n] * np.conj(SphHarmonic(n, theta[num1], phi[num2]))
                Out[num1, num2] = D_nm.T.conj() @ p_nm

        # Convert to dB and clip values below -22 dB
        out = 20 * np.log10(np.abs(Out))
        out = out - np.max(out)
        x, y = np.where(out == np.max(out))
        out = np.clip(out, -17, None)
        source_pos = np.array([x, y])

    if method == "MUSIC":
        P_music = np.zeros((len(theta), len(phi)), dtype=complex)  # MUSIC spectrogram
        Bn = np.zeros(((N + 1) ** 2, 1), dtype=complex)
        for n in range(N + 1):
            Bn[n ** 2:(n + 1) ** 2] = bn[n]

        L = 1  # The number of sound source
        a_nm = p_nm / Bn
        S = np.dot(a_nm, a_nm.T.conj())

        # Calculate the eigen value and eigen vector
        D, V = np.linalg.eigh(S)

        # Sort the eigenvalue and get the index
        I = np.argsort(D)
        Y = np.diag(D)[I]

        # Calculate the noise subspace
        E = V[:, I[:-L]]
        # Setup output grid

        for num1 in range(len(theta)):
            for num2 in range(len(phi)):
                y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                for n in range(N + 1):
                    y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2]).T
                P_music[num1, num2] = 1 / (y_nm @ E @ E.T.conj() @ y_nm.T.conj())

        # Convert to dB and clip values below -22 dB
        out = 10 * np.log10(np.abs(P_music))
        out = out - np.max(out)
        x, y = np.where(out == np.max(out))
        source_pos = np.array([x, y])

    if method == "SHMLE":

        Out = np.zeros((len(theta), len(phi)), dtype=complex)

        for num1 in range(len(theta)):
            for num2 in range(len(phi)):
                D_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                for n in range(N + 1):
                    D_nm[n ** 2:(n + 1) ** 2] = bn[n] * SphHarmonic(n, theta[num1], phi[num2]).conj()
                D_nm = D_nm[:, np.newaxis]
                Out[num1, num2] = np.linalg.norm(p_nm - D_nm @ np.linalg.pinv(D_nm) @ p_nm)

        # Convert to dB and clip values below -22 dB
        out = -20 * np.log10(np.abs(Out))
        out = out - np.max(out)
        x, y = np.where(out == np.max(out))
        source_pos = np.array([x,y])

    print("source_pos:", source_pos)
    plot_SSL_results(out, [Theta_l], [Phi_l], plot_method, method,"unit",sphere_config, vmin_value=None, vmax_value=None, source_est=source_pos.T)
    return out, source_pos

    
def setRoom(room_dim, mic_arrays_car, source_pos_car_list, signal_list, typ, rt60_tgt=None):
    """
    The function `setRoom` sets up a room with specified dimensions, microphone arrays, source
    positions, and signals, and simulates the room acoustics either in an anechoic or reverberant
    environment.
    
    :param room_dim: The dimensions of the room in meters (length, width, height)
    :param mic_arrays_car: The `mic_arrays_car` parameter is a list of microphone array positions in
    Cartesian coordinates. Each element in the list represents the position of a microphone array. The
    position of each microphone array is represented as a 2D array, where each row represents the x, y,
    and z coordinates of a
    :param source_pos_car_list: The `source_pos_car_list` parameter is a list of source positions in
    Cartesian coordinates. Each element in the list represents the position of a source in 3D space
    :param signal_list: The `signal_list` parameter is a list of audio signals that will be used as the
    source signals in the simulation. Each element in the list represents a different source signal
    :param typ: The "typ" parameter specifies the type of room simulation to be performed. It can have
    two possible values: "Anechoic" or "Reverb"
    :param rt60_tgt: The parameter "rt60_tgt" represents the target reverberation time (RT60) for the
    room. RT60 is a measure of how quickly sound decays in a room, and it is commonly used to
    characterize the level of reverberation in a space. In this function, if the
    :return: two values: `room` and `rt60_est`.
    """
    if typ == "Anechoic":
        for array_id, mic_pos in enumerate(mic_arrays_car):
            room = pra.AnechoicRoom(fs=24000)
            mic_pos = mic_pos.transpose()
            room.add_microphone_array(mic_pos)
            for i, (source_pos_car, signal) in enumerate(zip(source_pos_car_list, signal_list)):
                room.add_source(source_pos_car, signal=signal, delay=0)
            room.simulate()
            output_dir = '/content/Classical-Sound-Source-Localization-Algorithms-in-Spherical-Domain/Anechoic/Array_output_{}/'.format(array_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i in range(len(source_pos_car_list)):
                room.mic_array.to_wav(os.path.join(output_dir, 'source{}.wav'.format(i)), norm=True, bitdepth=np.float32)
        return room, None

    elif typ == "Reverb":
        for array_id, mic_pos in enumerate(mic_arrays_car):
            e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
            room = pra.ShoeBox(room_dim, fs=24000, materials=pra.Material(e_absorption), max_order=max_order)
            mic_pos = mic_pos.transpose()
            room.add_microphone_array(mic_pos)
            for i, (source_pos_car, signal) in enumerate(zip(source_pos_car_list, signal_list)):
                room.add_source(source_pos_car, signal=signal, delay=0)
            room.simulate()
            output_dir = '/content/Classical-Sound-Source-Localization-Algorithms-in-Spherical-Domain/Reverberant/Array_output_{}/'.format(array_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i in range(len(source_pos_car_list)):
                room.mic_array.to_wav(os.path.join(output_dir, 'source{}.wav'.format(i)), norm=True, bitdepth=np.float32)
        # Estimate the real T60 using the pyroomacoustics function
        rt60_est = np.mean(room.measure_rt60())  # get the average value for all frequency bands
        return room, rt60_est
    



def ssl_SHmethod_broad(mic_signals, fs, mic_pos_sph, Theta_l, Phi_l, method, sphere_config, plot_method, resolution, num_sources=1):
    """
    This script is stored some classical SSL algorithms in the spherical domain for source localization
    :param mic_signals: The received signals from microphone array
    :param fs: Sampling frequency
    :param mic_pos_sph:  The spherical coordinate of microphone array
    :param Theta_l:  The elevation of the source
    :param Phi_l:  The azimuth of the source
    :param method:  The chosen algorithm
    :param sphere_config: The configuration of sphere
    :param plot_method: The method of plotting (2D or 3D)
    :param resolution: The resolution for the grid in the space
    :param num_sources: The number of sound sources
    :return:  A figure, estimate azimuth , estimate elevation
    """
    # Transform the inputs to lists
    if not isinstance(Theta_l, list):
        Theta_l = [Theta_l]
    if not isinstance(Phi_l, list):
        Phi_l = [Phi_l]
    radius = 0.042
    c = 343       # Velocity of sound
    # num_mics = mic_pos_sph.shape[0]
    num_mics = 32
    K = 2048  # The length of signal frame
    Mic_Theta = mic_pos_sph[:,1]
    Mic_Phi = mic_pos_sph[:,2]
    signal_single_channel = mic_signals[:, 0]
    signal_type, freq = detect_signal_type(signal_single_channel, fs,0.9)
    theta = np.arange(0, np.pi, resolution / 180 * np.pi)
    phi = np.arange(0, 2 * np.pi, resolution / 180 * np.pi)
    x = []
    y = []

    if signal_type == 'Single':
        N = 1
        ka = 2 * np.pi * 500 / c * radius

        FrameNumber = mic_signals.shape[0] // K
        # label the effective data
        FrameFlag = np.zeros(FrameNumber + 1)
        count = 0
        flags = []
        for num in range(FrameNumber + 1):
            # determine this frame if is effective
            if np.sum(mic_signals[(num - 1) * K:num * K, 0] ** 2) > 10 * 1e-5:
                FrameFlag[num] = 1
                count += 1
                # Append the beginning position of effective frame
                flags.append((num - 1) * K + 1)
        # Calculate bn(ka)
        if sphere_config == "rigid":
            bn = np.zeros(N+1, dtype=complex)
            for n in range(N+1):
                jn_ka = sp.spherical_jn(n, ka)
                jn_ka_der = sp.spherical_jn(n, ka, derivative=True)
                yn_ka = sp.spherical_yn(n, ka)
                yn_ka_der = sp.spherical_yn(n, ka, derivative=True)

                # Compute the second kind of Hankel function and its derivative
                hn2_ka = jn_ka - 1j * yn_ka
                hn2_ka_der = jn_ka_der - 1j * yn_ka_der

                bn[n] = 4 * np.pi * 1j ** n * (jn_ka - jn_ka_der / hn2_ka_der * hn2_ka)

        if sphere_config == "open":
            bn = np.zeros(N+1, dtype=complex)
            for n in range(N+1):
                bn[n] = 4 * np.pi * 1j ** n * sp.spherical_jn(n, ka)

        # Calculate spherical harmonics
        Y_nm = np.zeros(((N+1)**2, num_mics), dtype=complex)
        for num in range(num_mics):
            for n in range(N + 1):
                Y_nm[n ** 2:(n + 1) ** 2, num] = SphHarmonic(n, Mic_Theta[num], Mic_Phi[num])

        # Define the signal in the time domain and freqeuncy domain
        x_p = np.zeros((K, num_mics), dtype=complex)
        X = np.zeros((K, num_mics), dtype=complex)

        for m in range(num_mics):
            x_p[:, m] = mic_signals[flags[0]:(flags[0] + K), m]
            X[:, m] = np.fft.fft(x_p[:, m])
        X_half = X[:K // 2, :]  # Keep only first half of spectrum

        I = int(np.floor(500 / fs * K)) + 1# Spectral line positions corresponding to single frequencies


        # Do spherical harmonic transform
        p_nm = 4 * np.pi / num_mics * Y_nm.conj() @ X_half.T
        if method == 'SHMVDR':
            sphCOV = np.dot(p_nm, p_nm.T.conj())
            lambda_reg = 1e-2  # 正则化因子，需要根据你的应用进行调整
            sphCOV += lambda_reg * np.eye(sphCOV.shape[0])
            # print(sphCOV.shape)
            out = []
            P_mvdr = np.zeros((len(theta), len(phi)), dtype=complex)  # MUSIC谱
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    U_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        U_nm[n ** 2:(n + 1) ** 2] = bn[n] * SphHarmonic(n, theta[num1], phi[num2]).conj()
                    U_nm = U_nm[:, np.newaxis]
                    invA_b = np.linalg.solve(sphCOV, U_nm)
                    b_invA_b = np.dot(U_nm.T.conj(), invA_b)
                    w_mvdr = invA_b / b_invA_b
                    P_mvdr[num1, num2] = w_mvdr.T.conj() @ sphCOV @ w_mvdr

            # Convert to dB and clip values below -22 dB
            out = 10 * np.log10(np.abs(P_mvdr))
            out = out - np.max(out)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]

        if method == "SHMUSIC":
            Bn = np.zeros(((N + 1) ** 2, 1), dtype=complex)
            for n in range(N + 1):
                Bn[n ** 2:(n + 1) ** 2] = bn[n]
            a_nm = p_nm / Bn
            S = np.dot(a_nm, a_nm.T.conj())
            # Calculate the Eigenvalue and Eigenvector
            D, V = np.linalg.eigh(S)
            # Sort the eigenvalue and get the index
            i = np.argsort(D)
            Y = np.diag(D)[i]
            # Calculate the noise subspace
            E = V[:, i[:-num_sources]]
            P_music = np.zeros((len(theta), len(phi)), dtype=complex)  # MUSIC谱
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2]).T
                    P_music[num1, num2] = 1 / (y_nm @ E @ E.T.conj() @ y_nm.T.conj())

            # Convert to dB and clip values below -22 dB
            out = 10 * np.log10(np.abs(P_music))
            out = out - np.max(out)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]

        if method == "SHMLE":
            out = []
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    D_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        D_nm[n ** 2:(n + 1) ** 2] = bn[n] * SphHarmonic(n, theta[num1], phi[num2]).conj()
                    D_nm = D_nm[:, np.newaxis]
                    Out[num1, num2] = np.linalg.norm(p_nm - D_nm @ np.linalg.pinv(D_nm) @ p_nm)

            out = -20 * np.log10(np.abs(Out))
            out = out - np.max(out)
            x, y = np.where(out == np.max(out))
            out = np.clip(out, -10, None)
            indices = []
            out_copy = np.copy(out)
            for _ in range(num_sources):
                max_index = np.argmax(out_copy)
                indices.append(np.unravel_index(max_index, out_copy.shape))
                out_copy[indices[-1]] = -np.inf

            # Convert the indices to the desired resolution
            source_positions = [(i * resolution, j * resolution) for i, j in indices]

        plot_SSL_results(out, Theta_l, Phi_l, plot_method, method, signal_type, sphere_config, vmin_value=None,
                         vmax_value=None, source_est=source_positions)
        return out, source_positions



    if signal_type == "Multiple":
        N = 4
        # Calculate the range of frequency: ka
        freq_up = round(K*c*N/(fs*2*np.pi*radius))
        freq_low = round(freq_up/2)+1

        ## divide the received signal into frames
        FrameNumber = mic_signals.shape[0] // K
        # label the effective data
        FrameFlag = np.zeros(FrameNumber + 1)
        count = 0
        flags = []
        for num in range(FrameNumber + 1):
            # determine this frame if is effective
            if np.sum(mic_signals[(num - 1) * K:num * K, 0] ** 2) > 10 * 1e-5:
                FrameFlag[num] = 1
                count += 1
                # Append the beginning position of effective frame
                flags.append((num - 1) * K + 1)

        if sphere_config == "rigid":
            # Calculate Bn
            bn = np.zeros(((N + 1) ** 2, freq_up), dtype=complex)

            for k in range(freq_up):
                ka = 2 * np.pi * k / K * fs / c * radius
                for n in range(N + 1):
                    jn_ka = sp.spherical_jn(n, ka)
                    jn_ka_der = sp.spherical_jn(n, ka, derivative=True)
                    yn_ka = sp.spherical_yn(n, ka)
                    yn_ka_der = sp.spherical_yn(n, ka, derivative=True)

                    # Compute the second kind of Hankel function and its derivative
                    hn2_ka = jn_ka - 1j * yn_ka
                    hn2_ka_der = jn_ka_der - 1j * yn_ka_der

                    bn[n ** 2:(n + 1) ** 2, k] = 4 * np.pi * (1j) ** n * (jn_ka - jn_ka_der / hn2_ka_der * hn2_ka)

        if sphere_config == "open":
            # Calculate Bn
            bn = np.zeros(((N + 1) ** 2, freq_up), dtype=complex)

            for k in range(freq_up):
                ka = 2 * np.pi * k / K * fs / c * radius
                for n in range(N + 1):
                    bn[n ** 2:(n + 1) ** 2, k] = 4 * np.pi * (1j) ** n * sp.spherical_jn(n, ka)

        # Calculate spherical harmonics
        Y_nm = np.zeros(((N+1)**2, num_mics), dtype=complex)
        for n in range(N+1):
            for m in range(num_mics):
                Y_nm[n ** 2:(n + 1) ** 2, m] = SphHarmonic(n, Mic_Theta[m], Mic_Phi[m])

        # Define the signal in the time domain and freqeuncy domain
        # 将信号变换到频域
        x_p = np.zeros((K, num_mics), dtype=complex)
        X = np.zeros((K, num_mics), dtype=complex)
        for m in range(num_mics):
            x_p[:, m] = mic_signals[flags[10]:(flags[10] + K), m]
            X[:, m] = np.fft.fft(x_p[:, m])

        # Do spherical transform
        p_nm = np.zeros((freq_up, (N + 1) ** 2), dtype=complex)
        for k in range(int(freq_low-1),int(freq_up)):
            p_nm[k, :] = 4 * np.pi / num_mics * X[k+1, :].dot(Y_nm.T.conj())
        p_nm = p_nm.T


        if method == "PWD":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            B_n = np.zeros(((N + 1) ** 2), dtype=complex)
            D_nm = np.zeros(((N + 1) ** 2, 1), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    for n in range(N + 1):
                        D_nm[n ** 2:(n + 1) ** 2, 0] = SphHarmonic(n, theta[num1], phi[num2])
                    for k in range(int(freq_low - 1), int(freq_up)):
                        a_nm = np.diag(1 / bn[:, k]) @ p_nm[:, k]
                        temp += np.linalg.norm(D_nm.T @ a_nm)
                    Out[num1, num2] = temp
            # Convert to dB and clip values below -22 dB
            out = 20 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]

        if method == "DAS":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    D_nm = np.zeros(((N + 1) ** 2, 1), dtype=complex)
                    for n in range(N + 1):
                        D_nm[n ** 2:(n + 1) ** 2, 0] = SphHarmonic(n, theta[num1], phi[num2]).conj()
                    for k in range(int(freq_low - 1), int(freq_up)):
                        d_nm = np.diag(bn[:, k]) @ D_nm
                        temp += np.linalg.norm(d_nm.T.conj() @ p_nm[:, k])
                    Out[num1, num2] = temp
            # Convert to dB and clip values below -22 dB
            out = 20 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]


        if method == "SHMVDR":
            lambda_reg = 1e-2
            P_mvdr = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2]).conj()
                    for k in range(int(freq_low - 1), int(freq_up)):
                        U_nm = np.diag(bn[:, k]) @ y_nm
                        U_nm = U_nm[:, np.newaxis]
                        sphCOV = np.outer(p_nm[:,k], p_nm[:,k].T.conj())
                        Lambda_reg = 1e-2
                        sphCOV += lambda_reg * np.eye(sphCOV.shape[0])
                        invA_b = np.linalg.solve(sphCOV , U_nm)
                        b_invA_b = np.dot(U_nm.T.conj(), invA_b)
                        w_mvdr = invA_b / b_invA_b
                        temp += np.linalg.norm(w_mvdr.T.conj() @ sphCOV @ w_mvdr)
                    P_mvdr[num1, num2] = temp

            # Convert to dB and clip values below -22 dB
            out = 10 * np.log10(np.abs(P_mvdr))
            out = out - np.max(out)
            out = np.clip(out, -10, None)

            indices = []
            out_copy = np.copy(out)
            for _ in range(num_sources):
                max_index = np.argmax(out_copy)
                indices.append(np.unravel_index(max_index, out_copy.shape))
                out_copy[indices[-1]] = -np.inf

            # Convert the indices to the desired resolution
            source_positions = [(i * resolution, j * resolution) for i, j in indices]


        if method == "SHMUSIC":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    y_nm = np.zeros(((N + 1) ** 2), dtype=complex)
                    for n in range(N + 1):
                        y_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2])
                    for k in range(int(freq_low - 1), int(freq_up)):
                        a_nm = np.diag(1 / bn[:, k]) @ p_nm[:, k]
                        S = np.outer(a_nm, a_nm.T.conj())
                        D, V = np.linalg.eigh(S)
                        i = np.argsort(D)
                        E = V[:, i[:-num_sources]]
                        temp += np.linalg.norm(y_nm.T @ E @ E.T.conj() @ y_nm.conj())
                    Out[num1, num2] = temp

            # Convert to dB and clip values below -22 dB
            out = -10 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)

            indices = []
            out_copy = np.copy(out)
            for _ in range(num_sources):
                max_index = np.argmax(out_copy)
                indices.append(np.unravel_index(max_index, out_copy.shape))
                out_copy[indices[-1]] = -np.inf

            # Convert the indices to the desired resolution
            source_positions = [(i * resolution, j * resolution) for i, j in indices]

        if method == "SHMLE":
            Out = np.zeros((len(theta), len(phi)), dtype=complex)
            for num1 in range(len(theta)):
                for num2 in range(len(phi)):
                    temp = 0
                    P_nm = np.zeros((N + 1) ** 2, dtype=complex)
                    for n in range(N + 1):
                        P_nm[n ** 2:(n + 1) ** 2] = SphHarmonic(n, theta[num1], phi[num2])
                    for k in range(int(freq_low - 1), int(freq_up)):
                        d_nm = np.diag(bn[:, k]) @ np.conj(P_nm)
                        d_nm = d_nm[:, np.newaxis]
                        temp += np.linalg.norm(p_nm[:, k] - d_nm @ np.linalg.pinv(d_nm) @ p_nm[:, k]) ** 2
                    Out[num1, num2] = temp

            # Convert to dB and clip values below -22 dB
            out = -10 * np.log10(np.abs(Out))
            out = out - np.max(out)
            out = np.clip(out, -10, None)
            x, y = np.where(out == np.max(out))
            source_positions = [(i * resolution, j * resolution) for i, j in zip(x, y)]


        plot_SSL_results(out, Theta_l, Phi_l, plot_method, method, 'Real Data', sphere_config, vmin_value=None,
                              vmax_value=None, source_est=source_positions)
        return out, source_positions
    

def calculate_spherical_coordinates(source_position, mic_array_position):
    """
    This function is used to calculate the spherical coordinate of source corresponding to different microphone array
    :param source_position:  The cartesian position of source (x, y, z)
    :param mic_array_position: The cartesian position of microphone array (x, y, z)
    :return: (r, theta, phi): Spherical coordinate of source corresponding to microphone array (r, theta, phi)
    """
    # Calculate the relative position between microphone array and source
    relative_position = source_position - mic_array_position
    # Calculate the distance
    r = np.linalg.norm(relative_position)

    # Elevation theta
    theta = np.arccos(relative_position[2] / r) # relative_position[2] 是 z 分量

    # Azimuth phi
    phi = np.arctan2(relative_position[1], relative_position[0]) # relative_position[0] 和 relative_position[1] 是 x 和 y 分量

    if phi < 0:
        phi += 2 * np.pi

    return r, theta, phi


def setRoom_multi(room_dim, mic_pos_car, source_pos_car_list, signal_list, typ, rt60_tgt=None):
    """
    Set anechoic room or normal room
    :param room_dim: The dimensions of the room
    :param mic_pos_car: The Cartesian coordinates of the microphone array
    :param source_pos_car_list: The list of Cartesian coordinates of sound sources
    :param signal_list: The list of signals
    :param typ: The type of room
    :param rt60_tgt: Target T60 for normal room
    :return: room
    """
    mic_pos_car = mic_pos_car.transpose()
    if typ == "Anechoic":
        room = pra.AnechoicRoom(fs=24000)
        room.add_microphone_array(mic_pos_car)
        for i, (source_pos_car, signal) in enumerate(zip(source_pos_car_list, signal_list)):
            room.add_source(source_pos_car, signal=signal, delay=0)
        room.simulate()
        # output
        for i in range(len(source_pos_car_list)):
            room.mic_array.to_wav('/content/Classical-Sound-Source-Localization-Algorithms-in-Spherical-Domain/Anechoic/Array_output{}.wav'.format(i), norm=True, bitdepth=np.float32)
        return room, None
    if typ == "Reverb":
        # Invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
        # Create the room
        room = pra.ShoeBox(room_dim, fs=24000, materials=pra.Material(e_absorption), max_order=max_order)
        # Place the sources in the room
        for i, (source_pos_car, signal) in enumerate(zip(source_pos_car_list, signal_list)):
            room.add_source(source_pos_car, signal=signal, delay=0)
        # Place the array in the room
        room.add_microphone_array(mic_pos_car)
        # Run the simulation
        room.simulate()
        # output
        for i in range(len(source_pos_car_list)):
            room.mic_array.to_wav(
                '/content/Classical-Sound-Source-Localization-Algorithms-in-Spherical-Domain/Reverberant/Array_output{}.wav'.format(i), norm=True, bitdepth=np.float32)
        # Estimate the real T60 using the pyroomacoustics function
        rt60_est = np.mean(room.measure_rt60())  # get the average value for all frequency bands
    return room, rt60_est
