import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import regularizers
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy
import gsd.hoomd
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# FILE PATHS
original_path = os.getcwd()
gsd_path = original_path + '/GSD'
clean_path = original_path + '/Clean'
model_path = original_path + '/Good_Model/07_12_good_model'
visual_path = original_path + '/Visuals'
xyz_path = original_path + '/XYZ'

# FIGURE SPECIFICS
font = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}
resolution = 500


def clean_data(file_name, output_file_name, num_frames, n_particles):
    """
    Description: Puts multiple particle datasets into one dataframe and loads into CSV

    Inputs:
        - training_file_name: name of file with raw data
        - output_file_name: name of file to dump cleaned data
        - num_frames: number of frames to be read
        - n_particles: number of particles
    Outputs:
        - dumps cleaned data to output_file_name
    """
    print(f'Cleaning {file_name}')
    os.chdir(gsd_path)
    with gsd.fl.open(file_name, mode = 'r') as f:
        file = gsd.hoomd.HOOMDTrajectory(f)
        all_frames = []
        for time_step in tqdm(range(num_frames)):
            frame = file.read_frame(time_step)
            single_frame = []
            for j in range(n_particles):
                single_frame.append(frame.particles.position[j][0])
                single_frame.append(frame.particles.position[j][1])
                single_frame.append(frame.particles.position[j][2])
            all_frames.append(single_frame)
    all_data = pd.DataFrame(all_frames)
    all_data = find_nearest_neighbors(all_data, n_particles)

    os.chdir(clean_path)
    all_data.to_csv(output_file_name, index=False)
    return

def find_nearest_neighbors(data, n_particles):
    '''
    Description: Finds two nearest neighbors for every particle into df

    Inputs:
        - data(df): dataset to analyze
        - n_particles: number of particles
    Outputs:
        - data_neighbors: df with 2 neighbor columns per particle
    '''
    data_neighbors = data.copy()
    data_neighbors.columns = data_neighbors.columns.astype(int)
    training_distances = [0]*len(data_neighbors)
    for i in tqdm(range(len(data_neighbors))):
        frame_list = [0]*5
        for j in range(0, n_particles):
            k=j*3
            single_particle=list(data_neighbors.iloc[i][k:k+3])
            frame_list[j] = single_particle
        distances = [0]*n_particles*2
        for j in range(n_particles):
            tree = scipy.spatial.KDTree(frame_list)
            dist,_ = tree.query(frame_list[j],k=3)
            if dist[1] >= 10000:
                dist[1] = 20000 - dist[1]
            if dist[2] >= 10000:
                dist[2] = 20000 - dist[2]
            distances[2*j] = dist[1]
            distances[2*j+1] = dist[2]
        training_distances[i] = distances
    training_distances = pd.DataFrame(training_distances)

    for i in range(0, n_particles*5, 5):
        particle_num = i//5
        insert_index = i//5
        data_neighbors.insert(i+3, str(particle_num)+'neighbor1', 
                              training_distances[insert_index])
        insert_index += 1
        data_neighbors.insert(i+4, str(particle_num)+'neighbor2', 
                              training_distances[insert_index])
        insert_index += 1
    
    data_neighbors.columns = data_neighbors.columns.astype(str)
    return data_neighbors

def split_into_sequences(dataset, n_samples):
    '''
    Description: Splits dataset into context windows for LSTM model

    Inputs:
        - dataset(df): data to be split
        - n_samples: context window length
    Outputs:
        - x(np array): x data in specified sequences
        - y(np.array): target data in specified sequences
    '''
    x, y = list(), list()
    for i in range(len(dataset)):
        end_point = i + n_samples
        if end_point <= len(dataset)-1:
            seq_x = dataset.values[i:end_point]
            seq_y = dataset.values[end_point]
            x.append(seq_x)
            y.append(seq_y)
    return np.array(x), np.array(y)

def plot_xyz(n_particles, test_data, pred, file_name, end):
    """
    Description: Plot xyz positions for each particle

    Inputs:
        - n_particles: num of particles
        - test_data: experimental results
        - pred: model predictions
        - file_name: file to save figure to
        - end: range of values to plot
    Outputs:
        - plot of xyz positions for each particle
    """

    os.chdir(visual_path)
    x = range(len(test_data))

    fig, axes = plt.subplots(n_particles, 3, figsize=(12, 8))

    cols = ['{}-Direction'.format(col) for col in ['X', 'Y', 'Z']]
    rows = ['Particle {}'.format(row) for row in range(1, n_particles + 1)]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontdict=font)
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, fontdict=font, rotation=0)

    for i in range(0, n_particles):
        test_col = i * 3
        pred_col = i * 5
        for j in range(3):
            test_col_index = test_col + j
            axes[i, j].plot(x[:end], test_data[str(test_col_index)][:end], 'bo', markersize=3, label='Test Data')
            pred_col_index = pred_col + j
            axes[i, j].plot(x[:end], pred[pred_col_index][:end], 'rx', markersize=3, label='Predicted Data')

            axes[i, j].grid(True, linestyle='--', linewidth=0.5)
            legend = axes[i, j].legend(loc='upper right', prop={'size':8})
            frame = legend.get_frame()
            frame.set_linewidth(0.5)

    fig.tight_layout()
    plt.savefig(file_name, dpi=resolution)
    plt.show()
    return

def plot_losses(history, key):
    """
    Description: Plot training and validation MSE

    Inputs:
        - history: results from model.fit
        - key: mean_squared_error
    Outputs:
        - plot of training and validation MSE vs epochs
    """

    current_datetime = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    plt.figure(figsize=(8, 6))
    plt.xlabel('Epochs', fontdict=font)
    plt.ylabel('Mean-Squared-Error', fontdict=font)
    plt.title('Training and Validation MSE Loss', fontdict=font)

    plt.plot(history.history[key], label=key, color='blue', linewidth=2)
    plt.plot(history.history['val_' + key], label='val_' + key, color='red', linewidth=2)

    plt.legend(prop=font)

    figure = plt.gcf()
    figure.set_size_inches(8, 6)

    plt.savefig(current_datetime + 'loss.jpg', dpi=resolution)
    plt.show()
    return

def dump_xyz(pred, file_name, n_particles):
    print('Dumping xyz data')
    os.chdir(xyz_path)
    rows = pred.shape[0]

    with open(file_name, 'w') as f:
        f.write('5\n')
        for i in tqdm(range(rows)):
            for j in range(0, n_particles*5, 5):
                x = pred[j][i]
                y = pred[j+1][i]
                z = pred[j+2][i]
                f.write(f'{j//3}\t{x}\t{y}\t{z}\n')
    print('Data copied to xyz file')


# FILE HANDLING
current_datetime = (datetime.now()).strftime("%Y_%m_%d_%H:%M:%S")

training_file_name = 'good_data_with_particle_interactions'   # file name without .gsd
clean_training_file_name = training_file_name + '_clean'

validation_file_name = 'short_sim_dt_10k'
clean_val_file_name = validation_file_name + '_clean'

testing_file_name = '2023-07-11_10k_hoomd_data'
clean_testing_file_name = testing_file_name + '_clean'

should_clean_training_data = False      # True: clean dataset
should_clean_val_data = False
should_clean_testing_data = False


# MODEL PARAMETERS
train_model = True                     # True: retrain model
N_particles = 5
data_size = N_particles*5
end_range_for_plot = 5000
training_frames = 10**6
val_frames = 10**4
testing_frames = 10**6
params = {'learning rate': 0.001,
        'hidden1 units': 128,
        'hidden2 units': 128,
        'epochs': 50,
        'batch size': 128,
        'n steps': 5,
        'dropout': 0.25}


# DATA PREPROCESSING
print('Loading data')
scaler = MinMaxScaler(feature_range=(-1,1))
if train_model:
    if should_clean_training_data:
        print('Train data preprocessing')
        clean_data(training_file_name + '.gsd',
                clean_training_file_name,
                training_frames,
                N_particles)
    os.chdir(clean_path)
    training_data = pd.read_csv(clean_training_file_name)

    training_data_scaled = pd.DataFrame(scaler.fit_transform(training_data[:len(training_data)-1]),
                                        columns=training_data.columns)
    x_train, y_train = split_into_sequences(training_data_scaled, params['n steps'])
    
    if should_clean_val_data:
        print('Val data preprocessing')
        clean_data(validation_file_name + '.gsd',
                clean_val_file_name, 
                val_frames,
                N_particles)
    os.chdir(clean_path)
    val_data = pd.read_csv(clean_training_file_name)

    x_val = pd.DataFrame(scaler.fit_transform(val_data[:len(val_data)-1]), 
                        columns=val_data.columns)
    x_val, y_val = split_into_sequences(x_val, params['n steps'])

if should_clean_testing_data:
    print('Test data preprocessing')
    clean_data(testing_file_name + '.gsd',
            clean_testing_file_name, 
            testing_frames, 
            N_particles)
os.chdir(clean_path)
testing_data = pd.read_csv(clean_testing_file_name)

x_test = pd.DataFrame(scaler.fit_transform(testing_data[:len(testing_data)-1]), 
                    columns=testing_data.columns)
x_test, y_test = split_into_sequences(x_test, params['n steps'])


# MODEL BUILDING
model = keras.Sequential()
model.add(LSTM(units=params['hidden1 units'], 
            input_shape=(params['n steps'], data_size),
            return_sequences=True))
model.add(LSTM(units=params['hidden2 units']))
model.add(Dropout(params['dropout']))
model.add(Dense(units=params['hidden2 units'],
                activation='relu',
                kernel_regularizer=regularizers.L2(0.05)))
model.add(Dense(data_size))
model.compile(loss='mean_squared_error',
            optimizer=Adam(learning_rate=params['learning rate']),
            metrics=['mean_squared_error'])
print(f'Model built with {model.count_params()} parameters')


# MODEL TRAINING
if train_model:
    print('Training model')
    history = model.fit(x_train,
                        y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch size'],
                        validation_data=(x_val, y_val))
    print('Finished training')
    model.save(model_path)
    print(f'Model saved into {model_path}')


# MODEL TESTING
model = keras.models.load_model(model_path)
print('Testing model')
predictions = pd.DataFrame(model.predict(x_test))
print('Evaluating test')
print(model.evaluate(x_test, y_test))
unscaled_predictions = pd.DataFrame(scaler.inverse_transform(predictions))
print(testing_data)
print(unscaled_predictions)
print('Finished testing')


# VISUALIZATION
viz_filename = (current_datetime+'_'+
                str(params['n steps'])+'steps_'+
                str(params['epochs'])+'epochs.jpg')
plot_xyz(N_particles, testing_data, unscaled_predictions, viz_filename, end_range_for_plot)

if train_model:
    plot_losses(history, 'mean_squared_error')
os.chdir(original_path)


# XYZ OUTPUT FILE
xyz_filename = current_datetime + testing_file_name + '.xyz'
dump_xyz(unscaled_predictions, xyz_filename, N_particles)


# TESTING WITH SINGLE DATA POINT
input_size = 64
epochs = 100
predictions = []
input = x_test[:input_size]
for i in range(epochs):
    output = model.predict(input)
    output=np.reshape(output,(input_size,1,25))
    predictions.append(output)
    input = output
predictions = np.array(predictions)
predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1],1,25))

first = []
for i in range(len(predictions)):
    for j in range(len(predictions[0])):
        first.append(predictions[i][j][0])

two = []
for i in range(len(predictions)):
    two.append(x_test[i][0][0])

plt.xlim([0,64])
plt.plot(range(len(first)), first, label='Generated X-Trajectory', color='blue', linewidth=2)
plt.plot(range(len(two)), two, label='Original X-Trajectory', color='red', linewidth=2)

plt.xlabel('Timesteps', fontdict=font)
plt.ylabel('X-Position', fontdict=font)
plt.title('Generating X-Trajectory', fontdict=font)

figure = plt.gcf()
figure.set_size_inches(8, 6)

plt.legend(prop={'size': 10})
os.chdir(visual_path)
plt.savefig(current_datetime + 'traj.png', dpi=resolution)
plt.show()