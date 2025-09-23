import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import chirp, fftconvolve
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.signal import convolve
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import toeplitz


#hejj

fs, lyd_data = wavfile.read(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\Signe_sang.wav")

lyd_data=list(np.array(lyd_data[0:fs*1])/max(lyd_data))

#plt.plot(lyd_data)
#plt.show()

print(f"Sample rate: {fs} Hz, length: {len(lyd_data):.2f}")

# ----------------------
# Room & array settings
# ----------------------
fs = 16000                   # sample rate
room_dim = [6.0, 3.0, 3.5]   # meters (L x W x H)
absorption = 0.2             # global absorption (0..1). Higher => less reverberant
max_order = 10               # image source order (higher -> more reflections)
t60_target = None            # optional: you can compute absorption from target T60

# Create a shoebox room
room = pra.ShoeBox(
    room_dim,
    fs=fs,
    materials=pra.Material(absorption),
    max_order=max_order,
)

# ----------------------
# Speakers (sources)
# ----------------------
# positions: [x,y,z]
sources = [
    [0.5, 0.0, 1.5],  
    [1.0, 0.0, 1.5],  
    [1.5, 0.0, 1.5],
    [2.0, 0.0, 1.5],
    [2.5, 0.0, 1.5],
    [3.0, 0.0, 1.5],
    [3.5, 0.0, 1.5],
    [4.0, 0.0, 1.5],
    [4.5, 0.0, 1.5],
    [5.0, 0.0, 1.5],
    [5.5, 0.0, 1.5],
]

for s in sources:
    room.add_source(s)

# ----------------------
# Microphones (microphone array) - Expanded with more microphones
# ----------------------
# Create a dense grid of microphones throughout the room
mic_positions = []

# Bright zone microphones (left side of room: x from 0.5 to 2.5)
for x in np.linspace(0.5, 2.5, 6):  # 6 microphones along x-axis
    for y in np.linspace(0.5, 2.5, 4):  # 4 microphones along y-axis
        mic_positions.append([x, y, 1.5])

# Dark zone microphones (right side of room: x from 3.5 to 5.5)
for x in np.linspace(3.5, 5.5, 6):  # 6 microphones along x-axis
    for y in np.linspace(0.5, 2.5, 4):  # 4 microphones along y-axis
        mic_positions.append([x, y, 1.5])


# Convert to numpy array
mic_positions = np.array(mic_positions).T

# Define which microphones are in bright and dark zones
bright_zone_mics = list(range(0, 24))  # First 24 mics (6x4 grid in bright zone)
dark_zone_mics = list(range(24, 48))   # Next 24 mics (6x4 grid in dark zone)
#center_line_mics = list(range(48, 60)) # Last 12 mics (center line)

from mpl_toolkits.mplot3d import Axes3D

def visualize_placement(room_dim, sources, mic_positions, bright_zone_mics, dark_zone_mics=None, center_line_mics=None):
    """
    Visualize the room layout with speakers and microphones
    
    Parameters:
    room_dim: list [L, W, H] - room dimensions
    sources: list of source positions
    mic_positions: numpy array of microphone positions (3 x N)
    bright_zone_mics: list of indices for bright zone microphones
    dark_zone_mics: list of indices for dark zone microphones (optional)
    center_line_mics: list of indices for center line microphones (optional)
    """
    
    fig = plt.figure(figsize=(15, 5))
    
    # Create 3 subplots: XY plane, XZ plane, and 3D view
    ax1 = fig.add_subplot(131)  # XY plane (top view)
    ax2 = fig.add_subplot(132)  # XZ plane (side view)
    ax3 = fig.add_subplot(133, projection='3d')  # 3D view
    
    # Plot room boundaries
    L, W, H = room_dim
    room_corners_xy = [[0, 0], [L, 0], [L, W], [0, W], [0, 0]]
    room_corners_xz = [[0, 0], [L, 0], [L, H], [0, H], [0, 0]]
    
    ax1.plot(*zip(*room_corners_xy), 'k-', linewidth=2, label='Room Boundary')
    ax2.plot(*zip(*room_corners_xz), 'k-', linewidth=2, label='Room Boundary')
    
    # Plot 3D room boundaries
    for z in [0, H]:
        ax3.plot([0, L, L, 0, 0], [0, 0, W, W, 0], [z, z, z, z, z], 'k-', linewidth=1, alpha=0.5)
    for x in [0, L]:
        ax3.plot([x, x, x, x, x], [0, W, W, 0, 0], [0, 0, H, H, 0], 'k-', linewidth=1, alpha=0.5)
    for y in [0, W]:
        ax3.plot([0, L, L, 0, 0], [y, y, y, y, y], [0, 0, H, H, 0], 'k-', linewidth=1, alpha=0.5)
    
    # Plot sources (speakers)
    sources_array = np.array(sources)
    ax1.scatter(sources_array[:, 0], sources_array[:, 1], c='red', s=100, marker='^', label='Speakers', edgecolors='black')
    ax2.scatter(sources_array[:, 0], sources_array[:, 2], c='red', s=100, marker='^', edgecolors='black')
    ax3.scatter(sources_array[:, 0], sources_array[:, 1], sources_array[:, 2], c='red', s=100, marker='^', label='Speakers', edgecolors='black')
    
    # Plot microphones
    mic_array = mic_positions.T
    
    # Bright zone microphones
    bright_mics = mic_array[bright_zone_mics]
    ax1.scatter(bright_mics[:, 0], bright_mics[:, 1], c='blue', s=50, marker='o', label='Bright Zone Mics', alpha=0.8)
    ax2.scatter(bright_mics[:, 0], bright_mics[:, 2], c='blue', s=50, marker='o', alpha=0.8)
    ax3.scatter(bright_mics[:, 0], bright_mics[:, 1], bright_mics[:, 2], c='blue', s=50, marker='o', label='Bright Zone Mics', alpha=0.8)
    
    # Dark zone microphones (if provided)
    if dark_zone_mics is not None:
        dark_mics = mic_array[dark_zone_mics]
        ax1.scatter(dark_mics[:, 0], dark_mics[:, 1], c='orange', s=50, marker='s', label='Dark Zone Mics', alpha=0.8)
        ax2.scatter(dark_mics[:, 0], dark_mics[:, 2], c='orange', s=50, marker='s', alpha=0.8)
        ax3.scatter(dark_mics[:, 0], dark_mics[:, 1], dark_mics[:, 2], c='orange', s=50, marker='s', label='Dark Zone Mics', alpha=0.8)
    
    # Center line microphones (if provided)
    if center_line_mics is not None:
        center_mics = mic_array[center_line_mics]
        ax1.scatter(center_mics[:, 0], center_mics[:, 1], c='green', s=50, marker='D', label='Center Line Mics', alpha=0.8)
        ax2.scatter(center_mics[:, 0], center_mics[:, 2], c='green', s=50, marker='D', alpha=0.8)
        ax3.scatter(center_mics[:, 0], center_mics[:, 1], center_mics[:, 2], c='green', s=50, marker='D', label='Center Line Mics', alpha=0.8)
    
    # Add zone boundaries
    ax1.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='Zone Boundary')
    ax1.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7, label='Zone Boundary')
    ax2.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set labels and titles
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY Plane)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (XZ Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D View')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary information
    print(f"Room dimensions: {room_dim[0]}m x {room_dim[1]}m x {room_dim[2]}m")
    print(f"Number of speakers: {len(sources)}")
    print(f"Number of microphones: {mic_positions.shape[1]}")
    if dark_zone_mics is not None:
        print(f"Bright zone microphones: {len(bright_zone_mics)}")
        print(f"Dark zone microphones: {len(dark_zone_mics)}")
    if center_line_mics is not None:
        print(f"Center line microphones: {len(center_line_mics)}")

visualize_placement(room_dim, sources, mic_positions, bright_zone_mics, dark_zone_mics)

room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

# ----------------------
# Compute RIRs
# ----------------------
room.compute_rir()  # fills room.rir [mic_index][source_index] -> array

n_mics = mic_positions.shape[1]
n_srcs = len(sources)

IR = room.rir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class ILZ_CNN(nn.Module):
    def __init__(self, M, S, L, T):
        super(ILZ_CNN, self).__init__()
        self.M, self.S, self.L, self.T = M, S, L, T

        self.conv1 = nn.Conv2d(1, 48, kernel_size=(M, 1))
        self.conv2 = nn.Conv2d(48, 24, kernel_size=(1, S))

        self.fc1 = nn.Linear(24, 10)  # <-- FIXED
        self.fc2 = nn.Linear(10, S * T)
        self.activation = nn.Hardtanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        q = x.view(-1, self.S, self.T)
        return q

J = 512  # filter length
model = ILZ_CNN(n_mics, n_srcs, len(IR[0][0]), J)

try:
    model.load_state_dict(torch.load(r'C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\SZC_model.pth'))
except FileNotFoundError:
    print("Model file not found. Proceeding without loading.")

x = torch.randn(1, 1, n_mics, n_srcs)   # shape: (batch, channels, M, S)
q_opt = model(x)


def pressure_field_2d(room_dim, sources, q_opt, lyd_data, grid_res=50, z_plane=1.5, J=J, fs=16000):
    """
    Compute and plot a 2D grid of sound pressure (SPL) in the room at a fixed z-plane,
    using the speakers with the applied filters. Also prints average pressure in bright and dark zones.
    Accepts q_opt as either a torch tensor or numpy array.
    """
    L = len(sources)

    # Ensure q_opt is a numpy array of shape (L, J)
    if isinstance(q_opt, torch.Tensor):
        q_opt = q_opt.detach().cpu().numpy()
        if q_opt.ndim == 3:  # (batch, L, J)
            q_opt = q_opt[0]
    q_matrix = q_opt.reshape(L, J)

    # 2D grid at fixed z
    x = np.linspace(0, room_dim[0], grid_res)
    y = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full_like(X, z_plane)

    pressure_field = np.zeros_like(X, dtype=float)
    test_signal = np.array(lyd_data[:fs//2])  # Use a short segment for speed

    for i in range(grid_res):
        for j in range(grid_res):
            point = [X[i, j], Y[i, j], Z[i, j]]
            h_point = []
            for src_idx, src_pos in enumerate(sources):
                r = np.linalg.norm(np.array(point) - np.array(src_pos))
                delay = int(r * fs / 343)
                h = np.zeros(J + 256)
                if delay < len(h):
                    h[delay] = 1 / (r + 1e-6)
                h_point.append(h)
            p = 0
            for l in range(L):
                filtered = lfilter(q_matrix[l], 1, test_signal)
                out_l = fftconvolve(filtered, h_point[l])
                p += np.sqrt(np.mean(out_l**2))  # RMS pressure
            pressure_field[i, j] = p

    pressure_dB = 20 * np.log10(pressure_field / (np.max(pressure_field) + 1e-12) + 1e-12)

    # Define bright and dark zones (left/right halves)
    bright_mask = X < room_dim[0] / 2
    dark_mask = X >= room_dim[0] / 2
    
    avg_bright = np.mean(pressure_field[bright_mask])
    avg_dark = np.mean(pressure_field[dark_mask])


    print(f"Average pressure in bright zone: {avg_bright:.4f}")
    print(f"Average pressure in dark zone: {avg_dark:.4f}")
    print(f"Pressure ratio (bright/dark) [dB]: {20 * np.log10(avg_bright / (avg_dark + 1e-12)):.2f} dB")

    plt.figure(figsize=(8, 6))
    plt.imshow(pressure_dB.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], aspect='auto', cmap='inferno')
    plt.colorbar(label='SPL [dB]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Sound Pressure Level at z={z_plane} m')
    plt.scatter([s[0] for s in sources], [s[1] for s in sources], c='cyan', marker='*', s=100, label='Speakers')
    plt.legend()
    plt.tight_layout()
    plt.show()

# After training, visualize the result (use fine grid)
pressure_field_2d(room_dim, sources, q_opt.detach().cpu().numpy(), lyd_data, grid_res=50, z_plane=1.5, J=J, fs=fs)


