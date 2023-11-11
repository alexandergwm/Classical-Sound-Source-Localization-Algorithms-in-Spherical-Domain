import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import matplotlib.cm as cm
import librosa
import librosa.display

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

"""
plot_tools: This script contains several tools for plotting, i.e. in time domain, frequency domain, STFT domain
"""
def plot_wave(signal, fs):
    """
    Plot the wave in time domain

    :param signal: signal in time domain
    :param fs: sampling rate of signal
    """
    time = np.arange(len(signal)) / fs
    # plot the wave
    plt.figure()
    plt.plot(time, signal)
    plt.title('Waveform in time domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_waveFD(signal, fs):
    """
    Plot the wave in frequency domain

    :param signal: signal in time domain
    :param fs: sampling rate of signal
    """
    # Compute the FFT
    fft_result = np.fft.rfft(signal)
    # Compute the frequencies associated with the FFT
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    # plot the FFT
    plt.figure()
    # p_ref = 20e-6  # reference pressure
    fft_result_dB = 20 * np.log10(np.abs(fft_result))
    plt.plot(freqs, fft_result_dB)
    plt.title('Waveform in frequency domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel("SPL(dB)")
    plt.tight_layout()
    plt.show()

def plot_stft(signals, fs):
    """
    The function `plot_stft` takes in a signal and its sampling rate, computes the Short-Time Fourier
    Transform (STFT), and plots the magnitude spectrogram.
    
    :param signals: The signals parameter is the input audio signal that you want to analyze using the
    Short-Time Fourier Transform (STFT). It can be a 1-dimensional array representing the audio waveform
    :param fs: The parameter "fs" represents the sampling rate of the audio signal. It is the number of
    samples per second in the audio signal
    """
    # Transfer from time domain to stft domain
    D = librosa.stft(signals)
    # Decompose it to the magnitude and phase
    magnitude, phase = librosa.magphase(D)
    # plot the STFT
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=fs, x_axis='time', y_axis='log')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()

    
def plot_grid(grid_points):
    """
    Plot a 3D grid in space

    :param grid_points: A 2D array of points [num_points,3]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_option(room, mic_arrays_car, source_pos_car_list):
    """
    Plot the microphone array in the room
    :param room:
    :param mic_arrays_car: The list of cartesian positions of microphone arrays
    :param source_pos_car_list: The list of cartesian positions of sound sources
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the room
    for wall in room.walls:
        corners = wall.corners.T
        ax.plot(corners[[0, 1, 2, 3, 0], 0], corners[[0, 1, 2, 3, 0], 1], corners[[0, 1, 2, 3, 0], 2], 'k')

    # Plot the microphone array
    for i, mic_pos_car in enumerate(mic_arrays_car):
        ax.scatter(mic_pos_car[:, 0], mic_pos_car[:, 1], mic_pos_car[:, 2], c='g', marker='o',
               label=f'Microphone Array {i+1}')
    # Plot the source locations
    for i, source_pos_car in enumerate(source_pos_car_list):
        ax.scatter(source_pos_car[0], source_pos_car[1], source_pos_car[2], c='r', marker='x', label=f'Source {i+1}', s=100)
    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()


def plot_source_positions(source_DoAs):
    """
    Plot the directions of arrival (DoAs) of multiple sound sources on a 2D plot with theta on y-axis and phi on x-axis.
    :param source_DoAs: A numpy array of shape (n, 2) containing the DoAs of the sound sources, with each row being a pair (theta, phi).
    """
    fig, ax = plt.subplots()

    # Set the axes labels
    ax.set_xlabel('Phi [deg]')
    ax.set_ylabel('Theta [deg]')

    # Set the axes limits
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 180])

    # Plot a green dot at the origin
    ax.plot(0, 0, 'go', markersize=10, label='Microphone Array')

    # Plot the points representing the DoAs of the sound sources
    for theta, phi in source_DoAs:
        theta = theta * 180 / np.pi
        phi = phi * 180 / np.pi
        ax.plot(phi, theta, 'rx', label='Source')

    # Add a legend
    ax.legend()
    plt.grid()
    # Show the plot
    plt.show()


def plot_microphone(mic_pos_car):
    """
    Plot the microphone array in the room
    :param mic_pos_car: The cartesian position of microphone array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the microphone array
    ax.scatter(mic_pos_car[:, 0], mic_pos_car[:, 1], mic_pos_car[:, 2], c='g', marker='o',
               label='Microphones')
    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    # Show the legend
    ax.legend()
    plt.show()

def verify_shapeofArray(mic_pos_car):
    # 创建3个子图，分别对应xy, xz, yz平面的投影
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # xy平面投影
    axs[0].scatter(mic_pos_car[:, 0], mic_pos_car[:, 1], c='g')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Projection onto XY plane')
    axs[0].set_aspect('equal', 'box')

    # xz平面投影
    axs[1].scatter(mic_pos_car[:, 0], mic_pos_car[:, 2], c='g')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].set_title('Projection onto XZ plane')
    axs[1].set_aspect('equal', 'box')

    # yz平面投影
    axs[2].scatter(mic_pos_car[:, 1], mic_pos_car[:, 2], c='g')
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')
    axs[2].set_title('Projection onto YZ plane')
    axs[2].set_aspect('equal', 'box')

    plt.show()




def plot_SHfunction(Ndec):
    # 计算 theta, phi 网格
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    # 对应的笛卡尔坐标
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # 创建3D图形
    index = 0
    for n in range(Ndec + 1):
        for m in range(-n, n + 1):
            Ynm = sph_harm(m, n, theta, phi)  # 取实部，或者可以取绝对值
            c = np.abs(Ynm)  # 使用平均值
            index += 1

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(
                x, y, z, rstride=1, cstride=1, facecolors=plt.cm.jet(c), alpha=0.5, linewidth=0)
            # add colorbar
            m = cm.ScalarMappable(cmap= cm.jet)
            m.set_array(c)
            fig.colorbar(m, shrink=0.5)
            plt.show()

def plot_grid(theta, phi):
    """
    This function is used to plot the grid that we create in spherical domain
    :param theta: elevation
    :param phi: azimuth
    :return:
    """

    # 将 theta 和 phi 转换为笛卡尔坐标
    r = 1
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # 创建一个新的图形和一个 3D 子图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建一个散点图来表示 theta 和 phi 网格
    ax.scatter(x, y, z)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()

def plot_MicArray(mic_pos_sph):
    """
    This script is used to plot the shape of spherical microphone array
    :param mic_pos_sph: The spherical coordinate of each microphone
    :return: A plot
    """
    R = mic_pos_sph[0,0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Nmic = len(mic_pos_sph)
    mic_pos_rad = mic_pos_sph[:,1:3]
    # 设置 3D 轴
    ax.quiver(0, 0, 0, 2*R, 0, 0, color='r')
    ax.text(2*R, 0, 0, 'x', color='r', fontsize=12)
    ax.quiver(0, 0, 0, 0, 2*R, 0, color='g')
    ax.text(0, 2*R, 0, 'y', color='g', fontsize=12)
    ax.quiver(0, 0, 0, 0, 0, 2*R, color='b')
    ax.text(0, 0, 2*R, 'z', color='b', fontsize=12)

    # 设置球体半径
    sphere_radius = 0.05 * R

    # 计算麦克风位置
    spheresX = R * np.cos(mic_pos_rad[:, 0]) * np.cos(mic_pos_rad[:, 1])
    spheresY = R * np.sin(mic_pos_rad[:, 0]) * np.cos(mic_pos_rad[:, 1])
    spheresZ = R * np.sin(mic_pos_rad[:, 1])

    # 绘制每个麦克风位置的小球体
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for i in range(Nmic):
        x = spheresX[i] + sphere_radius * np.cos(u) * np.sin(v)
        y = spheresY[i] + sphere_radius * np.sin(u) * np.sin(v)
        z = spheresZ[i] + sphere_radius * np.cos(v)
        ax.plot_surface(x, y, z, color='b')

        ax.text(1.1 * spheresX[i], 1.1 * spheresY[i], 1.1 * spheresZ[i], str(i + 1), color='w', fontsize=12)

    # 绘制大球体
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_surface(x, y, z, color='c', alpha=0.3)

    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    ax.axis('off')


def plot_SSL_results(out, Theta_l_rad_list, Phi_l_rad_list, plot_type, method, signal_type, sphere_config, vmin_value=None, vmax_value=None, source_est=None):
    """
    This script is used to plot SSL algorithm result
    :param out: The output of algorithm
    :param Theta_l_rad_list:  A list of the real sound source elevations
    :param Phi_l_rad_list:  A list of the real sound source azimuths
    :param plot_type:  The type of plotting
    :param method:  The name of algorithm
    :param signal_type:  The type of signal
    :param sphere_config: The configuration of the sphere
    :param vmin_value: The minimum spectrum
    :param vmax_value: The maximum spectrum
    :param source_est: The estimated source position
    :return: A plot
    """
    Theta_l_deg_list = [angle * 180 / np.pi for angle in Theta_l_rad_list]
    Phi_l_deg_list = [angle * 180 / np.pi for angle in Phi_l_rad_list]
    if plot_type == "2D":
        plt.figure(figsize=(10, 8), dpi=250)
        im = plt.imshow(out, origin='lower', extent=[0, 360, 0, 180], aspect='auto', cmap='jet', vmin=vmin_value,
                   vmax=vmax_value)
        cb = plt.colorbar(im, label='[dB]')
        cb.ax.tick_params(labelsize=16)  
        cb.set_label('[dB]', size=25)  
        plt.xlabel(r'$\phi$ [deg]', fontsize=25)
        plt.ylabel(r'$\theta$ [deg]', fontsize=25)
        plt.xticks(fontsize=18)  
        plt.yticks(fontsize=18)  
        for Theta_l_deg, Phi_l_deg in zip(Theta_l_deg_list, Phi_l_deg_list):
            plt.plot(Phi_l_deg, Theta_l_deg, 'bx', markersize=10, markeredgewidth=3, label='Real Source Position')
        if source_est is not None:
            for idx, est in enumerate(source_est):
                label = 'Estimated Source Position' if idx == 0 else ""  
                plt.plot(est[1], est[0], 'ro', markersize=10, fillstyle='none', markeredgewidth=3, label=label)
        plt.legend(loc='upper right', fontsize=16)
        plt.show()

    elif plot_type == "3D":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 360), np.linspace(0, np.pi, 180))
        phi_deg = np.rad2deg(phi)
        theta_deg = np.rad2deg(theta)

        surface = ax.plot_surface(phi_deg, theta_deg, out, cmap='jet')
        cb = fig.colorbar(surface, label='[dB]', shrink=0.5, aspect=5)
        cb.ax.tick_params(labelsize=16)  
        cb.set_label('[dB]', size=18)  
        ax.set_xlabel('Phi [deg]', fontsize=18)
        ax.set_ylabel(r'$\theta$ [deg]', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)  

        for Theta_l_deg, Phi_l_deg in zip(Theta_l_deg_list, Phi_l_deg_list):
            source = ax.scatter3D(Phi_l_deg, Theta_l_deg, np.max(out), color='ForestGreen', s=100, edgecolor='DarkBlue',
                                  linewidth=1.5, label='Real Source Position')
            ax.text(Phi_l_deg, Theta_l_deg, np.max(out),
                    '({:.1f}, {:.1f}, {:.1f})'.format(Phi_l_deg, Theta_l_deg, np.max(out)), color='ForestGreen',
                    fontsize=14)

        if source_est is not None:
            for idx, est in enumerate(source_est):
                label = 'Estimated Source Position' if idx == 0 else ""
                est_source = ax.scatter3D(est[1], est[0], np.max(out), color='DarkRed', s=100, edgecolor='DarkBlue',
                                          linewidth=1.5, label=label, marker='o', facecolors='none')

        plt.legend()
        plt.show()

    else:
        print("Invalid plot type. Please select either '2D' or '3D'.")

        
def plot_spherical_grid(resolution):
    """
    Plots a spherical grid with a given resolution.
    
    :param resolution: The resolution in degrees for the grid points
    """
    # Create meshgrid for the polar and azimuthal angles
    theta, phi = np.mgrid[0:np.pi:complex(0, 180/resolution), 0:2*np.pi:complex(0, 360/resolution)]

    # Convert polar and azimuthal angles to cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title for the subplot
    ax.set_title(f'Simulated grid for possible points in the space with grid resolution = {resolution}°')

    # Show the plot
    plt.show()


