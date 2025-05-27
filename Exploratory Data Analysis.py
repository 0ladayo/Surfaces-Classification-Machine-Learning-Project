# %% [markdown]
# ### Import the necessary libraries

# %%
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.signal import welch

# %% [markdown]
# ### Read the datasets

# %%
asphalt_df = pd.read_csv('./datasets/afw.csv') # Reading the asphalt dataset

concrete_df = pd.read_csv('./datasets/cfd.csv') # Reading the concrete dataset

grass_df = pd.read_csv('./datasets/gsw.csv') # Reading the grass dataset

stone_df = pd.read_csv('./datasets/scw.csv') # Reading the stone dataset

tile_df = pd.read_csv('./datasets/ts.csv') # Reading the tile dataset

# %% [markdown]
# In order to better explore the data, we will consolidate the 5 datasets as one.

# %%
df = pd.concat([asphalt_df, concrete_df, grass_df, stone_df, tile_df], axis = 0) # Joining the 5 datasets as one

df = df.reset_index(drop = True)

# %% [markdown]
# ### Exploratory Data Analysis

# %%
df.head(5) # Check for the first 5 rows

# %%
df.tail() # Check the last 5 rows

# %%
df.info()  # Retrieve information about the dataset (dataframe)

# %% [markdown]
# The dataset comprises seven features: three aceelerometer axes (x, y, z), three gyro axes (x, y, z), and the surface type. The first six features are floating-point numbers, while the surface feature is a string. Although no data appears to be missing, we will perform a verification to confirm this.

# %%
df.isna().sum() # Explicitly checking for missing data

# %% [markdown]
# There are no missing data in the dataset

# %%
df.surface.unique() # Checking for the unique data in the surface feature

# %% [markdown]
# The unique check on the surface feature confirmed the dataset contains data collected from only 5 distinct surfaces.

# %%
df.groupby('surface').describe() # Obtain statistical information of the numerical features

# %% [markdown]
# The above gives statistical information about the numerical features in the dataset for each surfaces in the dataset.

# %% [markdown]
# ### Distribution of the features for each surface - Histogram

# %%
plt.figure(figsize = (20, 15))

for i, feature in enumerate(df.columns[:6], 1):
    
    plt.subplot(2, 3, i)
    
    sns.histplot(data = df, x = feature, kde = True, hue = 'surface', element = 'step',
                
                palette = 'tab10')
    
    plt.title(f'Distribution of {feature}')
    
plt.tight_layout()

plt.show()

# %% [markdown]
# The distributions of each numerical feature for the different surfaces, as shown in the plots above are mostly symmetrical around their means. Additionally, the gyro features exhibit a much wider range than the accelerometer features

# %% [markdown]
# ### Box Plots

# %%
plt.figure(figsize = (15, 15))

for i, feature in enumerate(df.columns[:-1], 1):
    
    plt.subplot(3, 2, i)
    
    sns.boxplot(x = 'surface', y = feature, data = df)
    
    plt.title(f'Box plot of {feature}')
    
plt.tight_layout()

plt.show()

# %% [markdown]
# The box plots shows that the tile surface has the lowest interquartile range (IQR), suggesting less variability, potentially due to its smoothness. Conversely, the grass surface exhibits the highest IQR, indicating greater variability, whcih could be attributed to its uneveness (slope).

# %% [markdown]
# ### Correlations
# 
# #### HeatMap

# %%
df_2 = df.drop(['surface'], axis = 1)

corr = df_2.corr()

plt.figure(figsize = (8, 6))

sns.heatmap(corr, annot = True, cmap = 'coolwarm', center = 0)

plt.title('Feature Correlation Heatmap')

plt.show()

# %% [markdown]
# The heatmap illustrates weak or no correlation between the dataset's features

# %% [markdown]
# #### Pairwise Relationships

# %%
sns.pairplot(df, hue = 'surface')

plt.suptitle('Pairwise Relationships', verticalalignment = 'top')

plt.show()

# %% [markdown]
# The pairwise plots suggest that none of the feature combinations distinctly classify the surface types
# 
# Since the goal of the project is for surface classifications rather directional surface classification , we are better off consolidating the accelerometer and gyro data (x, y, z) into a single absolute value

# %%
df_consolidated = df.copy() # make a copy of the initial dataframe

# %%
# create a new column accelerometer and gyro which is absolute value of the x, y, z of the accelerometer and gyro respectively

df_consolidated['accelerometer'] = np.sqrt(df['x accelerometer']**2 + df['y accelerometer']**2
                                        + df['z accelerometer']**2)

df_consolidated['gyro'] = np.sqrt(df['x gyro']**2 + df['y gyro']**2 + df['z gyro']**2)

# %%
#drop the x, y, z of the accelerometer and gyro columns

df_consolidated = df_consolidated.drop(['x accelerometer', 'y accelerometer', 'z accelerometer',
                                       'x gyro', 'y gyro', 'z gyro' ], axis = 1)

# Re-arrange the columns

df_consolidated = df_consolidated.iloc[:,[1, 2, 0]]

# %%
df_consolidated.head()

# %%
df_consolidated.groupby('surface').describe()

# %% [markdown]
# While there isn't so much difference in the mean of the accelerometer across the 5 surfaces, extracting the mean and max of the gyro could help in surface classifications

# %% [markdown]
# ### Time Plots

# %%
plt.figure(figsize = (20, 15))

names_list = ['asphalt', 'concrete', 'grass', 'stones', 'tile']

for index, name in enumerate(names_list):

    plt.subplot(3, 2, index + 1)

    i_df = df_consolidated[df_consolidated['surface'] == name].reset_index(drop = True)

    plt.plot(i_df['accelerometer'], label = 'accelerometer')

    plt.title(f'Time Plot - {name}')

    plt.xlabel('Time (s) / 100')

    plt.ylabel('Acceleration')

    plt.legend()

# %% [markdown]
# The time plot for asphlat shows a flat trend towards the end, suggesting a period of inactivity. This segement should be removed from the analysis.
# 
# The grass plot displays a distinct trend due to the surface'sloping nature, in contrast to the flatness of the oher surfaces.

# %%
asphalt_consolidated_df = df_consolidated[df_consolidated['surface'] == 'asphalt']

asphalt_consolidated_df2 = asphalt_consolidated_df[0:14850]

plt.figure(figsize = (20, 15))

plt.plot(asphalt_consolidated_df2['accelerometer'], label = 'accelerometer')

plt.title('Time Plot - Asphalt')

plt.xlabel('Time (s) / 100')

plt.ylabel('Acceleration')

plt.legend()

# %% [markdown]
# The flat surface in the asphalt time plot has been removed.

# %%
concrete_consolidated_df = df_consolidated[df_consolidated['surface'] == 'concrete'].reset_index(drop = True)

grass_consolidated_df = df_consolidated[df_consolidated['surface'] == 'grass'].reset_index(drop = True)

stone_consolidated_df = df_consolidated[df_consolidated['surface'] == 'stones'].reset_index(drop = True)

tile_consolidated_df = df_consolidated[df_consolidated['surface'] == 'tile'].reset_index(drop = True)

# %%
df_consolidated2 = pd.concat([asphalt_consolidated_df2, concrete_consolidated_df, grass_consolidated_df, 
                              
                              stone_consolidated_df, tile_consolidated_df], axis = 0)

# %%
df_consolidated2 = df_consolidated2.reset_index(drop = True)

df_consolidated2.head()

# %%
df_consolidated2.to_csv('./datasets/imu_data.csv', index = False)

# %% [markdown]
# ### Distribution of the features for each surface - Histogram

# %%
plt.figure(figsize = (20, 15))

for i, feature in enumerate(df_consolidated2.columns[:2], 1):
    
    plt.subplot(1, 2, i)
    
    sns.histplot(data = df_consolidated2, x = feature, kde = True, hue = 'surface', element = 'step',
                
                palette = 'tab10')
    
    plt.title(f'Distribution of {feature}')
    
plt.tight_layout()

plt.show()

# %% [markdown]
# The distributions of the accelerometer for the different surfaces, as shown in the plots above are mostly symmetrical around their means while that of the gyro exhibit right skewed behaviour across the different surfaces. Additionally, the gyro features exhibit a much wider range than the accelerometer features.
# 
# Also, the asphalt surface exhibit a lower heavy tailed pattern in both the accelerometer and gyro compared to the other surface and extracting the kurtosis during feature extraction could help in differentiating the asphalt surfaces from the other surfaces. Extracting the skewness could help play a part in differenting the surfaces also.

# %% [markdown]
# ### Box Plots

# %%
plt.figure(figsize = (15, 10))

for i, feature in enumerate(df_consolidated2.columns[:2], 1):
    
    plt.subplot(1, 2, i)
    
    sns.boxplot(x = 'surface', y = feature, data = df_consolidated2)
    
    plt.title(f'Box plot of {feature}')
    
plt.tight_layout()

plt.show()

# %% [markdown]
# The box plots shows that the tile surface has the lowest interquartile range (IQR) in both the accelerometer and gyro, suggesting less variability, likely due to its smoothness. Utilizing standard deviation as a feature during feature extraction could effectively differentiate tile from other surfaces

# %% [markdown]
# ### Correlations
# 
# #### HeatMap

# %%
df_consolidated_2 = df_consolidated2.drop(['surface'], axis = 1)

corr = df_consolidated_2.corr()

plt.figure(figsize = (8, 6))

sns.heatmap(corr, annot = True, cmap = 'coolwarm', center = 0)

plt.title('Feature Correlation Heatmap')

plt.show()

# %% [markdown]
# The heatmap illustrates weak or no correlation between the accelerometer and the gyro

# %% [markdown]
# #### Pairwise Relationships

# %%
sns.pairplot(df_consolidated2, hue = 'surface')

plt.suptitle('Pairwise Relationships', verticalalignment = 'top')

plt.show()

# %% [markdown]
# The pairwise plots suggest that just plot of accelerometer vs the gyro or gyro vs the accelerometer doesn't distinctly classify the surface types

# %% [markdown]
# ### Frequency Domain
# 
# To analyze the time series in terms of frequency, we will employ the Fourier Transform to convert the features from the time domain to the frequency domain.

# %%
dataframe_list = [asphalt_consolidated_df2, concrete_consolidated_df, grass_consolidated_df, stone_consolidated_df, tile_consolidated_df]

names_list = ['Asphalt', 'Concrete', 'Grass', 'Stone', 'Tile']

fs = 100

plt.figure(figsize = (15, 10))

for index, (df, name) in enumerate(zip(dataframe_list, names_list)):

    fft_result= np.fft.fft(df['accelerometer'])

    fft_freq= np.fft.fftfreq(len(df['accelerometer']), 1/fs)

    positive_freqs = fft_freq > 0

    plt.subplot(3, 2, index + 1)
    
    plt.plot(fft_freq[positive_freqs], np.abs(fft_result[positive_freqs]))
    
    plt.title(f'Frequency Plot of Acceleration Signal - {name}')
    
    plt.xlabel('Frequency (Hz)')
    
    plt.ylabel('Amplitude')

plt.tight_layout()

plt.show()

# %% [markdown]
# The frequency plot of the acceleration data of the five surfaces shows high amplitude spikes at very low frequencies (0 HZ) which may suggest npise or the influence of gravitational acceleration data.
# 
# Additionally, each surface exhibits different amplitude peaks, with tile having the lowest and grass the highest. The concrete and asphalt have quite similar behavior (pattern). Extracting the peak amplitude during feature extraction can help to improve surface classifications.

# %%
plt.figure(figsize = (15, 10))

for index, (df, name) in enumerate(zip(dataframe_list, names_list)):
    
    fft_result= np.fft.fft(df['gyro'])

    fft_freq= np.fft.fftfreq(len(df['gyro']), 1/fs)

    positive_freqs = fft_freq > 0

    plt.subplot(3, 2, index + 1)
    
    plt.plot(fft_freq[positive_freqs], np.abs(fft_result[positive_freqs]))
    
    plt.title(f'Frequency Plot of Gyro Signal - {name}')
    
    plt.xlabel('Frequency (Hz)')
    
    plt.ylabel('Amplitude')

plt.tight_layout()

plt.show()

# %% [markdown]
# The gyroscopic frquency plot for for the five surfaces shows a flat amplitude at high frequencies typically indicative of linear motion. The high amplitude at the low frequencies is characterizing the different (angular) turning of the wheel of the rollator as we moved through the different surfaces.
# 
# The stone surface shows the greatest low-frequency amplitude, possibly due to its roughness. On the other hand, the tile surface exhbits the lowest amplitude at these frequencies, likely because of its smoothness.
# 
# The grass's slope might also contribute to its low-frequency amplitude observed in the plot.
# 
# Extracting the peak amplitude of the gyroscopic data into feature extraction could imporvethe surface classification.

# %%
plt.figure(figsize = (15, 20))

for index, (df, name) in enumerate(zip(dataframe_list, names_list)):

    signal = df['accelerometer']

    fs = 100

    frequencies, psd = welch(signal, fs, nperseg = 500)

    psd_dB = 10 * np.log10(psd)

    plt.subplot(3, 2, index + 1)

    plt.plot(frequencies, psd_dB)

    plt.title(f'Power Spectral Density of Acelerometer Signal - {name}')
    
    plt.xlabel('Frequency [Hz]')
    
    plt.ylabel('Power/Frequency [dB/Hz]')
    
    plt.grid(True)

plt.show()

# %%
plt.figure(figsize = (15, 20))

for index, (df, name) in enumerate(zip(dataframe_list, names_list)):

    signal = df['gyro']

    fs = 100

    frequencies, psd = welch(signal, fs, nperseg = 500)

    psd_dB = 10 * np.log10(psd)

    plt.subplot(3, 2, index + 1)

    plt.plot(frequencies, psd)

    plt.title(f'Power Spectral Density of Gyro Signal - {name}')
    
    plt.xlabel('Frequency [Hz]')
    
    plt.ylabel('Power/Frequency [V^2/Hz]')
    
    plt.grid(True)

plt.show()

# %%



