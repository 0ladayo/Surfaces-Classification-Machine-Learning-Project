# %% [markdown]
# ### Import the necessary libraries

# %%
import pandas as pd

import numpy as np

from scipy.signal import welch

from scipy.stats import skew, kurtosis

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, GridSearchCV

# %% [markdown]
# #### Read the dataset

# %%
df = pd.read_csv('./datasets/imu_data_filtered.csv')

# %% [markdown]
# #### Optimal Window Size and Kernel Determination
# 
# Support Vector Machines offers various kernels, and the window size can significantly impact the recognition accuracy. Therefore, we will use window sizes ranging from 100 to 500 to fit models with different kernels and evaluate their accuracy and F1 score

# %%
def sliding_window(df, window_size):
    
    groups = []
    
    current_group = []
    
    current_surface = None  

    for index, row in df.iterrows():
        
        if current_surface is None:
            
            current_surface = row['surface']  

        if row['surface'] == current_surface:
            
            current_group.append((row['filtered accelerometer'], row['gyro']))

            if len(current_group) == window_size:
                
                groups.append((current_group, current_surface)) 
                
                current_group = [] 

        else:
            
            if len(current_group) == window_size:
                
                groups.append((current_group, current_surface))

            current_group = [(row['filtered accelerometer'], row['gyro'])]
            
            current_surface = row['surface']
            
    if len(current_group) == window_size:
        
        groups.append((current_group, current_surface))

    return groups

# %%
def calculate_mean(data):
    
    accelerometer_mean = np.mean([x[0] for x in data[0]])
    
    gyro_mean = np.mean([x[1] for x in data[0]])
    
    return accelerometer_mean, gyro_mean

def calculate_std(data):
    
    accelerometer_std = np.std([x[0] for x in data[0]]) 
    
    gyro_std = np.std([x[1] for x in data[0]])
    
    return accelerometer_std, gyro_std

def calculate_max(data):
    
    accelerometer_max = np.max([x[0] for x in data[0]]) 
    
    gyro_max = np.max([x[1] for x in data[0]]) 
    
    return accelerometer_max, gyro_max

def calculate_min(data):
    
    accelerometer_min = np.min([x[0] for x in data[0]])
    
    gyro_min = np.min([x[1] for x in data[0]])
    
    return accelerometer_min, gyro_min

def calculate_iqr(data):
    
    accelerometer_iqr = np.percentile([x[0] for x in data[0]], 75) - np.percentile([x[0] for x in data[0]], 25)
    
    gyro_iqr = np.percentile([x[1] for x in data[0]], 75) - np.percentile([x[1] for x in data[0]], 25) 
    
    return accelerometer_iqr, gyro_iqr

def calculate_skewness(data):
    
    accelerometer_skew = skew([x[0] for x in data[0]])
    
    gyro_skew = skew([x[1] for x in data[0]])
    
    return accelerometer_skew, gyro_skew

def calculate_kurtosis(data):
    
    accelerometer_kurt = kurtosis([x[0] for x in data[0]])
    
    gyro_kurt = kurtosis([x[1] for x in data[0]])
    
    return accelerometer_kurt, gyro_kurt
    
def calculate_fft_peak(data, sampling_rate):
    
    fft_acc = np.fft.fft([x[0] for x in data[0]])
    
    fft_gyro = np.fft.fft([x[1] for x in data[0]])
    
    mag_acc = np.abs(fft_acc)
    
    mag_gyro = np.abs(fft_gyro)
    
    peak_index_acc = np.argmax(mag_acc)
    
    peak_index_gyro = np.argmax(mag_gyro)
    
    peak_amplitude_acc = mag_acc[peak_index_acc]
    
    peak_amplitude_gyro = mag_gyro[peak_index_gyro]

    freqs = np.fft.fftfreq(len(fft_acc), d = 1/sampling_rate)

    peak_frequency_acc = np.abs(freqs[peak_index_acc])
    
    return peak_amplitude_acc, peak_frequency_acc, peak_amplitude_gyro

def calculate_psd(data):
    
    accelerometer_data = np.array([x[0] for x in data[0]])
    
    gyro_data = np.array([x[1] for x in data[0]])

    f_acc, psd_acc = welch(accelerometer_data, fs=100, nperseg=100)
    
    f_gyro, psd_gyro = welch(gyro_data, fs=100, nperseg=100)

    peak_freq_acc = f_acc[np.argmax(psd_acc)]
    
    peak_power_acc = np.max(psd_acc)
    
    peak_freq_gyro = f_gyro[np.argmax(psd_gyro)]
    
    peak_power_gyro = np.max(psd_gyro)

    peak_power_acc_db = 10*np.log10(peak_power_acc)

    peak_power_gyro_db = 10*np.log10(peak_power_gyro)

    return peak_freq_acc, peak_power_acc_db, peak_freq_gyro, peak_power_gyro_db

def extract_surface(data):

    surface = data[1] 

    return surface

# %%
def extract_features(segments):
    
    features_list = []
    
    for segment in segments:

        psd_features = calculate_psd(segment)
        
        features = {
            'accelerometer_mean': calculate_mean(segment)[0],
            'accelerometer_std': calculate_std(segment)[0],
            'accelerometer_min': calculate_min(segment)[0],
            'accelerometer_max': calculate_max(segment)[0],
            'accelerometer_iqr': calculate_iqr(segment)[0],
            'accelerometer_skewness': calculate_skewness(segment)[0],
            'accelerometer_kurtosis': calculate_kurtosis(segment)[0],
            'accelerometer_fft_peak_amplitude': calculate_fft_peak(segment, 100)[0],
            'accelerometer_fft_peak_frequency': calculate_fft_peak(segment, 100)[1],
            'accelerometer_peak_power': psd_features[1],
            'accelerometer_peak_freq': psd_features[0],
            'gyro_mean': calculate_mean(segment)[1],
            'gyro_std': calculate_std(segment)[1],
            'gyro_min': calculate_min(segment)[1],
            'gyro_max': calculate_max(segment)[1],
            'gyro_iqr': calculate_iqr(segment)[1],
            'gyro_skewness': calculate_skewness(segment)[1],
            'gyro_kurtosis': calculate_kurtosis(segment)[1],
            'gyro_fft_peak_amplitude': calculate_fft_peak(segment, 100)[2],
            'gyro_peak_power': psd_features[3],
            'gyro_peak_freq': psd_features[2],
            'surface': extract_surface(segment)
        }
        features_list.append(features)
    
    return features_list

# %%
window_size_list = range(100, 600, 100)

kernel_list = ['linear', 'rbf', 'sigmoid']

# %%
#Initialize label encoder to convert categorical lables to numerical values for ML models

label_encoder = LabelEncoder()

# Initiate scaler to standardize input features

scaler = StandardScaler()

# Initialize an empty list to store window size, kernel, accuracy, and f1 score

accuracy_f1score_list = [] 

for i in window_size_list:

    # Generate windowed data from the dataframe based on the current window size

    window_data_list = sliding_window(df, i)

    # Extract features from each window into a list

    features_list = extract_features(window_data_list)

    # Create a dataframe from the features list

    features_df = pd.DataFrame(features_list)

    # Encode 'surface' column to numeric

    features_df['surface_encoded'] = label_encoder.fit_transform(features_df['surface'])

    # Drop the original 'surface' column

    features_df = features_df.drop('surface', axis = 1)

    # Assign feature columns to X 
    
    X = features_df.drop('surface_encoded', axis = 1)

    # Assign the encoded surface column to y

    y = features_df['surface_encoded']

    # Split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

    # Scale the training feature set

    X_train_scaled = scaler.fit_transform(X_train)

    # Scale the testing feature set
    
    X_test_scaled = scaler.transform(X_test)  

    for j in kernel_list:

         # Initialize an SVM classifier with the current kernel
        
        clf = svm.SVC(kernel = j,  random_state = 42)

        # Train the model with scaled training features and labels

        clf.fit(X_train_scaled, y_train)

        #  Predict labels for the scaled test features

        y_pred = clf.predict(X_test_scaled)

        # Calculate the accuracy of the predictions

        accuracy = accuracy_score(y_test, y_pred)

        # Calculate the weighted f1 score of the predictions

        f1 = f1_score(y_test, y_pred, average = 'weighted')

        # Store window size, kernel, accuracy, and f1 score in a dictionary

        accuracy_f1score_dic = {

        'window size': i,

        'kernel': j,

        'accuracy' : accuracy,

        'f1 score': f1

        }

        # Append the dictionary to the list

        accuracy_f1score_list.append(accuracy_f1score_dic)

# %%
# Create a DataFrame from the list containing dictionaries of window size, kernel, accuracy, and f1 score

accuracy_f1score_df = pd.DataFrame(accuracy_f1score_list) 

# %%
# Extract unique kernel types used in the models from the DataFrame 'accuracy_f1score_df'

unique_kernel_components = accuracy_f1score_df['kernel'].unique()

# Create a figure with specific dimensions

plt.figure(figsize = (10,6))

# Loop through each unique kernel type

for i in unique_kernel_components:

    # Filter the DataFrame for rows corresponding to the current kernel type

    kernel_data = accuracy_f1score_df[accuracy_f1score_df['kernel'] == i]

    # Extract the 'window size' data for plotting

    window_size = kernel_data['window size']

    # Extract the 'accuracy' data for plotting

    accuracy = kernel_data['accuracy']

    # Plot 'accuracy' against 'window size' for the current kernel type
    
    plt.plot(window_size, accuracy, label = f'Kernel : {i}')

# Label the x-axis as 'Window Size'

plt.xlabel('Window Size')

# Label the y-axis as 'Accuracy'

plt.ylabel('Accuracy')

# Set the title of the plot

plt.title('Accuracy vs Window Size by Kernel')

# Add a legend

plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

# Enable grid lines

plt.grid(True)

# Adjust layout

plt.tight_layout()

# Display the plot

plt.show()

# %%
# Extract unique kernel types used in the models from the DataFrame 'accuracy_f1score_df'

unique_kernel_components = accuracy_f1score_df['kernel'].unique()

# Create a figure with specific dimensions

plt.figure(figsize = (10,6))

# Loop through each unique kernel type

for i in unique_kernel_components:

    # Filter the DataFrame for rows corresponding to the current kernel type

    kernel_data = accuracy_f1score_df[accuracy_f1score_df['kernel'] == i]

    # Extract the 'window size' data for plotting

    window_size = kernel_data['window size']

    # Extract the 'f1_score' data for plotting

    f1_scores = kernel_data['f1 score']

    # Plot 'f1 score' against 'window size' for the current kernel type
    
    plt.plot(window_size, f1_scores, label = f'Kernel : {i}')

# Label the x-axis as 'Window Size'

plt.xlabel('Window Size')

# Label the y-axis as 'Accuracy'

plt.ylabel('F1 Score')

# Set the title of the plot

plt.title('F1 Score vs Window Size by Kernel')

# Add a legend

plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

# Enable grid lines

plt.grid(True)

# Adjust layout

plt.tight_layout()

# Display the plot

plt.show()

# %% [markdown]
# From the plots above, the highest accuracy and f1 score were obtained with a  window size of 300 and a linear kernel.
# 
# We will use these settings for our analysis moving forward

# %%
# Partition the IMU data into windows of size 300

window_data_list = sliding_window(df, 300)

# Extract features from each window into a list

features_list = extract_features(window_data_list)

# %%
# Create a DataFrame from the list containing the features

features_df = pd.DataFrame(features_list)

# %%
# Encode 'surface' column to numeric

features_df['surface_encoded'] = label_encoder.fit_transform(features_df['surface'])

# Drop the original 'surface' column

features_df = features_df.drop('surface', axis = 1)

 # Assign feature columns to X

X = features_df.drop('surface_encoded', axis = 1)

# Assign the encoded surface column to y

y = features_df['surface_encoded']

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

# %% [markdown]
# ##### Class Balance
# 
# For classification problems, it is important to maintain balanced classes in both the training and testing datasets. Although the 'stratify = y' argument helps achieve this balance, verifying it through visualization remains essential.

# %%
# Get the value count of each class in the label train set

y_train_value_counts = y_train.value_counts()

y_train_value_counts

# %%
# Create a dictionary to map numeric encoded surface types to their descriptive names

surface_mapping = {0: 'asphalt', 1: 'concrete', 2: 'grass', 3: 'stones', 4: 'tile'}

# Rename the index of y_train_value_counts DataFrame using the surface mapping

y_train_value_counts =  y_train_value_counts.rename_axis('surface_encoded').rename(surface_mapping)

# Plot the counts of each surface type as a bar chart

y_train_value_counts.plot(kind = 'bar')

# Set the x-axis ticks to have no rotation

plt.xticks(rotation = 0)

# Label the x-axis

plt.xlabel('Surfaces')

# Label the y-axis

plt.ylabel('Count')

# Set the title of the plot

plt.title('Count of Each Classes (Surfaces) in the Train Set')

# Display the plot

plt.show()

# %%
# Get the value count of each class in the label test set

y_test_value_counts = y_test.value_counts()

y_test_value_counts

# %%
# Rename the index of y_test_value_counts DataFrame using the surface mapping

y_test_value_counts =  y_test_value_counts.rename_axis('surface_encoded').rename(surface_mapping)

# Plot the counts of each surface type as a bar chart

y_test_value_counts.plot(kind = 'bar')

# Set the x-axis ticks to have no rotation

plt.xticks(rotation = 0)

# Label the x-axis

plt.xlabel('Surfaces')

# Label the y-axis

plt.ylabel('Count')

# Set the title of the plot

plt.title('Count of Each Classes (Surfaces) in the Test Set')

# Display the plot

plt.show()

# %% [markdown]
# #### Cross Validation

# %%
# Initialize an SVM classifier with a linear kernel

clf = svm.SVC(kernel = 'linear', random_state = 42)

# Create a Stratified K-Fold object with 5 splits, enabling shuffling to ensure random distribution of data in each fold

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

# Build a pipeline that first scales data using StandardScaler and then applies the SVM classifier

pipeline = make_pipeline(StandardScaler(), clf)

# Perform cross-validation using the defined stratified k-fold approach and pipeline and calculating accuracy for each fold

cv_scores = cross_val_score(pipeline, X_train, y_train, cv = skf, scoring = 'accuracy')

# Print the accuracy scores for each fold

print(cv_scores)

# Calculate and print the mean of the cross-validation accuracy scores

print(np.mean(cv_scores))

# %% [markdown]
# Using 5-fold cross-validation, the average accuracy achieved is 86.66%, with a variability of approximately 9.09%

# %%
# Perform cross-validation using the defined stratified k-fold approach and pipeline and calculating f1 score for each fold

f1_scores = cross_val_score(pipeline, X_train, y_train, cv = skf, scoring = 'f1_weighted')

# Print the f1 scores for each fold

print(f1_scores)

# Calculate and print the mean of the cross-validation F1 scores

print(np.mean(f1_scores))

# %% [markdown]
# Using 5-fold cross-validation, the average f1 score achieved is 86.60%, with a variability of approximately 9.13%

# %% [markdown]
# #### Testing the model on the Unseen data X_test

# %%
# Fit the pipeline to the training data

pipeline.fit(X_train, y_train)

# Predict the labels for the test data using the trained pipeline

y_pred = pipeline.predict(X_test)

# Calculate the accuracy of the predictions on the test data

test_accuracy = accuracy_score(y_test, y_pred)

# Calculate the F1 score of the predictions on the test data

test_f1_score = f1_score(y_test, y_pred, average = 'weighted')

# Print the test accuracy

print("Test Accuracy:", test_accuracy)

# Print the test F1 score

print("Test F1 Score:", test_f1_score)

# %% [markdown]
# We achieved a test accuracy and F1 score of 93.05% on the unseen data (X_test), indicating a good balance between precision and recall. Also shows the model is doing very well on unseen data
# 

# %%
# Define the class names for the labels in the dataset

class_names = ['asphalt', 'concrete', 'grass', 'stones', 'tiles']

# Compute the confusion matrix using true labels and predictions

cm = confusion_matrix(y_test, y_pred)

# Create a confusion matrix display object with specified labels and the calculated matrix

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)

# Plot the confusion matrix

disp.plot(cmap = plt.cm.Blues)

# Set the title

plt.title('Confusion Matrix')

# Display the plot

plt.show()

# %% [markdown]
# From the confusion matrix, one asphalt label was misclassified as concrete, two grass labels were misclassified as stones, and two stone labels were misclassified as grass. The concrete and tile labels were classified accurately.

# %% [markdown]
# ### Applying PCA
# 
# We are going to apply PCA choosing the number of components that explains at least 95% of the variance to our model

# %%
# Create a pipeline2 that first scales data, then applies PCA, and uses a classifier 'clf'

pipeline2 = make_pipeline(StandardScaler(), PCA(n_components = 0.95), clf)

# Perform cross-validation using the defined stratified k-fold approach and pipeline2 and calculating accuracy for each fold

cv_scores2 = cross_val_score(pipeline2, X_train, y_train, cv = skf, scoring = 'accuracy')

# Print the accuracy scores for each fold

print(cv_scores2)

# Calculate and print the mean of the cross-validation accuracy scores

print(np.mean(cv_scores2))

# %% [markdown]
# Applying PCA that capture at least 95% of the variance, the average accuracy achieved is 86.66%, with a variability of approximately 15.15%
# 
# We observed that, although the average accuracy remains the the same as without PCA, the variability has increased from approximately 9.09% to 15.15% indicating that the model has become less stable

# %%
# Perform cross-validation using the defined stratified k-fold approach and pipeline2 and calculating f1 score for each fold

f1_scores2 = cross_val_score(pipeline2, X_train, y_train, cv = skf, scoring = 'f1_weighted')

# Print the f1 scores for each fold

print(f1_scores2)

# Calculate and print the mean of the cross-validation F1 scores

print(np.mean(f1_scores2))

# %% [markdown]
# Applying PCA, the average F1 Score achieved is 86.66%, with a variability of approximately 15.47%
# 
# We observed that, although the average F1 score remains the the same as without PCA, the variability has increased from approximately 9.13% to 15.47% indicating that the model has become less stable

# %% [markdown]
# #### Testing the model on the Unseen data X_test

# %%
# Fit the pipeline2 to the training data

pipeline2.fit(X_train, y_train)

# Predict the labels for the test data using the trained pipeline

y_pred = pipeline2.predict(X_test)

# Calculate the accuracy of the predictions on the test data

test_accuracy = accuracy_score(y_test, y_pred)

# Calculate the F1 score of the predictions on the test data

test_f1_score = f1_score(y_test, y_pred, average = 'weighted')

# Print the test accuracy

print("Test Accuracy:", test_accuracy)

# Print the test F1 score

print("Test F1 Score:", test_f1_score)

# %% [markdown]
# We achieved a test accuracy of 87.5% and an F1 score of 87.44% on the unseen data (X_test), both lower than the results obtained without PCA. This decrease was expected, as applying PCA increased the variability in cross-fold accuracy and F1 scores

# %%
# Compute the confusion matrix using true labels and predictions

cm = confusion_matrix(y_test, y_pred)

# Create a confusion matrix display object with specified labels and the calculated matrix

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)

# Plot the confusion matrix

disp.plot(cmap=plt.cm.Blues)

# Set the title

plt.title('Confusion Matrix with PCA (95%)')

# Display the plot

plt.show()

# %% [markdown]
# From the confusion matrix, three asphalt label was misclassified as concrete, one concrete label was misclassified as asphalt, three grass labels were misclassified as stones, and two stone labels were misclassified as grass. The tile labels were classified accurately.

# %% [markdown]
# We are going to apply PCA choosing the number of components that explains at least 99% of the variance

# %%
# Create a pipeline3 that first scales data, then applies PCA, and uses a classifier 'clf'

pipeline3 = make_pipeline(StandardScaler(), PCA(n_components = 0.99), clf)

# Perform cross-validation using the defined stratified k-fold approach and pipeline2 and calculating accuracy for each fold

cv_scores3 = cross_val_score(pipeline3, X_train, y_train, cv = skf, scoring = 'accuracy')

# Print the accuracy scores for each fold

print(cv_scores3)

# Calculate and print the mean of the cross-validation accuracy scores

print(np.mean(cv_scores3))

# %% [markdown]
# Applying PCA that captures at least 99% of the variance, we achieved an average accuracy of 88.48% with a variability of approximately 12.12%. Although the average accuracy has increased compared to without PCA, the variability also rose from about 9.09% to 12.12%. This is an improvement over the results obtained with PCA capturing at least 95% of the variance.

# %%
# Perform cross-validation using the defined stratified k-fold approach and pipeline2 and calculating f1 score for each fold

f1_scores3 = cross_val_score(pipeline3, X_train, y_train, cv = skf, scoring = 'f1_weighted')

# Print the f1 scores for each fold

print(f1_scores3)

# Calculate and print the mean of the cross-validation F1 scores

print(np.mean(f1_scores3))

# %% [markdown]
# Applying PCA that captures at least 99% of the variance, we achieved an average F1 score of 88.49% with a variability of approximately 12.12%. Although the average F1 score increased compared to without PCA, the variability also rose from about 9.13% to 12.12%. This improvement is notable compared to results obtained with PCA capturing at least 95% of the variance.

# %% [markdown]
# #### Testing the model on the Unseen data X_test

# %%
# Fit the pipeline3 to the training data

pipeline3.fit(X_train, y_train)

# Predict the labels for the test data using the trained pipeline

y_pred = pipeline3.predict(X_test)

# Calculate the accuracy of the predictions on the test data

test_accuracy = accuracy_score(y_test, y_pred)

# Calculate the F1 score of the predictions on the test data

test_f1_score = f1_score(y_test, y_pred, average = 'weighted')

# Print the test accuracy

print("Test Accuracy:", test_accuracy)

# Print the test F1 score

print("Test F1 Score:", test_f1_score)

# %% [markdown]
# We achieved a test accuracy and F1 score of 93.05% on the unseen data (X_test), matching the metrics obtained without PCA and surpassing the results when PCA captured at least 95% of the variance. This indicates that our feature extraction was adequate and robust, as using PCA that captures at least 99% of the variance is necessary to achieve results comparable to those without PCA.

# %%
# Compute the confusion matrix using true labels and predictions

cm = confusion_matrix(y_test, y_pred)

# Create a confusion matrix display object with specified labels and the calculated matrix

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)

# Plot the confusion matrix

disp.plot(cmap=plt.cm.Blues)

# Set the title

plt.title('Confusion Matrix with PCA (99%)')

# Display the plot

plt.show()

# %% [markdown]
# From the confusion matrix, one asphalt label was misclassified as concrete, two grass labels were misclassified as stones, and two stone labels were misclassified as grass. The concrete and tile labels were classified accurately.

# %% [markdown]
# ### Applying HyperParameter Tuning

# %%
# Set the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__kernel': ['rbf', 'sigmoid', 'linear'],
    'svc__gamma': [0.1, 1, 'scale', 'auto']
}

# %%
# Set up a grid search with a pipeline3, parameter grid, and cross-validation method (Stratified K-Fold)

grid_search = GridSearchCV(pipeline3, param_grid = param_grid, cv = skf)

# Fit the grid search model to the training data to find the best parameters

grid_search.fit(X_train, y_train)

# Print the best parameters found by the grid search

print("Best parameters:", grid_search.best_params_)

# Print the highest accuracy achieved by the best performing model found using the grid search

print("Best SVM Accuracy: {:.2f}".format(grid_search.best_score_))

# %% [markdown]
# The best model accuracy we achieved from applying hyperparameter tuning is 88%.

# %% [markdown]
# #### Testing the model on the Unseen data X_test

# %%
# Use the best model found by the grid search to make predictions on the test data

grid_predictions = grid_search.predict(X_test)

# %%
# Calculate the accuracy of the predictions against the actual test labels

test_accuracy = accuracy_score(y_test, grid_predictions)

# Calculate the weighted F1 score of the predictions against the actual test labels

test_f1_score = f1_score(y_test, grid_predictions, average = 'weighted')

# Print the accuracy of the model on the test data

print("Test Accuracy:", test_accuracy)

# Print the F1 score of the model on the test data

print("Test F1 Score:", test_f1_score)

# %% [markdown]
# Using the best model from the hyperparameter tuning, we achieved a test accuracy and F1 score of 93.05%, matching the metrics obtained without PCA and hyperparameter tuning. Therefore, hyperparameter tuning did not improve our test accuracy.

# %%
# Compute the confusion matrix using true labels and predictions

cm = confusion_matrix(y_test, grid_predictions)

# Create a confusion matrix display object with specified labels and the calculated matrix

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)

# Plot the confusion matrix

disp.plot(cmap=plt.cm.Blues)

# Set the title

plt.title('Confusion Matrix with Hyperparameter Tuning')

# Display the plot

plt.show()

# %% [markdown]
# From the confusion matrix, one asphalt label was misclassified as concrete, two grass labels were misclassified as stones, and two stone labels were misclassified as grass. The concrete and tile labels were classified accurately.

# %%



