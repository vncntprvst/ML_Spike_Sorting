import time

from utils.model_initializer import *
from utils.data_preparation import *
from utils.model_predict import *
from utils.clustering import *
from utils.wandb_initializer import *
from utils.finetune_models import *
from utils.file_opener_raw_recording_data import *
from utils.filter_signal import *
from utils.spike_detection import *


from config_files.config_finetune import *
from config_files.config_pretraining import *
from config_files.config_data_preprocessing import *


class Run:
    """
    Class for running the full Spike sorting pipeline on raw MEA recordings (MCS file format) (if benchmark=False)
    or benchmarking the method on simulated spike files (if benchmark=True)

    This class integrates different components of the spike sorting pipeline, including
    data preparation, model initialization, pretraining, finetuning, prediction, clustering,
    and evaluation.

    Attributes:
        model_config: Configuration for the model.
        data_path: Path to the data.
        pretrain_method: Method for pretraining the model.
        fine_tune_method: Method for finetuning the model.
        pretraining_config: Configuration for pretraining.
        finetune_config: Configuration for finetuning.

    Methods:
        prepare_data(): Prepare the data for training and testing.
        initialize_model(): Initialize the machine learning model.
        initialize_wandb(): Initialize Weights & Biases for tracking experiments.
        pretrain(): Pretrain the model.
        finetune(): Finetune the model.
        predict(): Predict latent representations using the model.
        cluster_data(): Perform clustering on the encoded data.
        evaluate_spike_sorting(): Evaluate the results of spike sorting.
        execute_pretrain(): Execute the pretraining phase.
        execute_finetune(): Execute the finetuning phase.
    """

    def __init__(self, model_config, data_path, benchmark, pretrain_method, fine_tune_method):
        self.model_config = model_config
        self.data_path = data_path
        self.benchmark = benchmark
        self.pretrain_method = pretrain_method
        self.fine_tune_method = fine_tune_method
        self.data_preprocessing_config = Config_Preprocessing(self.data_path)
        self.pretraining_config = Config_Pretraining(self.data_path, self.model_config.MODEL_TYPE)
        self.finetune_config = Config_Finetuning(self.data_path, self.model_config.MODEL_TYPE)


    def extract_spikes_from_raw_recording(self):
        """Extract spikes from raw recording, using chunked processing for large files."""
        import os
        import h5py
        
        path = self.data_preprocessing_config.DATA_PATH
        
        # Check file size to decide on chunked vs full processing
        file_size_gb = os.path.getsize(path) / (1024**3) if os.path.exists(path) else 0
        
        # Use chunked processing for files > 10GB
        if file_size_gb > 10:
            print(f"[extract_spikes] Large file detected ({file_size_gb:.1f} GB). Using chunked processing...")
            self._extract_spikes_chunked()
        else:
            print(f"[extract_spikes] Small file ({file_size_gb:.1f} GB). Using standard processing...")
            recording_data, electrode_stream, fsample = file_opener_raw_recording_data(self.data_preprocessing_config)
            filtered_signal = filter_signal(recording_data, fsample, self.data_preprocessing_config)
            spike_file = spike_detection(filtered_signal, electrode_stream, fsample, self.data_preprocessing_config)

    def _extract_spikes_chunked(self):
        """Process large recordings in chunks to avoid OOM."""
        import h5py
        import pickle
        import math
        from utils.filter_signal import butter_bandpass, butter_bandpass_high, ellip_filter
        from scipy.signal import lfilter
        
        config = self.data_preprocessing_config
        path = config.DATA_PATH
        
        # Parameters
        chunk_duration_sec = config.INTERVAL_LENGTH  # Use same chunk size as spike detection (300s default)
        overlap_samples = 1000  # Overlap to avoid edge effects in filtering
        
        print(f"[chunked] Opening {path}")
        
        with h5py.File(path, 'r') as f:
            # Get metadata
            if 'sample_rate' in f.attrs:
                fsample = int(f.attrs['sample_rate'])
            elif 'sampling_frequency' in f.attrs:
                fsample = int(f.attrs['sampling_frequency'])
            else:
                raise ValueError("HDF5 file missing sample_rate attribute")
            
            dataset = f['recording']
            total_frames, num_channels = dataset.shape
            chunk_frames = int(chunk_duration_sec * fsample)
            
            print(f"[chunked] Recording: {total_frames} frames, {num_channels} channels, {fsample} Hz")
            print(f"[chunked] Duration: {total_frames/fsample/60:.1f} minutes")
            print(f"[chunked] Processing in {chunk_duration_sec}s chunks ({chunk_frames} frames)")
            
            # Create electrode stream for channel info
            electrode_stream = SimpleElectrodeStream(num_channels)
            ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
            
            # Spike detection parameters from config
            min_TH = config.MIN_TH
            max_TH = config.MAX_TH
            points_pre = config.DAT_POINTS_PRE_MIN
            points_post = config.DAT_POINTS_POST_MIN
            refrec_period = config.REFREC_PERIOD
            reject_ch = config.REJECT_CHANNELS or [None]
            
            # Get filter coefficients once
            frequencies = config.FREQUENCIES
            order = config.ORDER
            lowcut, highcut = frequencies[0], frequencies[1]
            from scipy.signal import butter
            nyq = 0.5 * fsample
            b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='bandpass')
            
            # Accumulate all spikes
            all_spikes = []
            spike_times_per_channel = [[0] for _ in range(num_channels)]
            
            num_chunks = math.ceil(total_frames / chunk_frames)
            
            for chunk_idx in range(num_chunks):
                start_frame = chunk_idx * chunk_frames
                # Add overlap at the end, but not past the file
                end_frame = min(start_frame + chunk_frames + overlap_samples, total_frames)
                actual_chunk_frames = end_frame - start_frame
                
                print(f"[chunked] Processing chunk {chunk_idx+1}/{num_chunks} (frames {start_frame}-{end_frame})")
                
                # Load chunk
                chunk_data = dataset[start_frame:end_frame, :]
                
                # Apply scale if present
                if 'scale_to_uV' in f.attrs:
                    chunk_data = chunk_data * f.attrs['scale_to_uV']
                
                # Filter the chunk
                filtered_chunk = lfilter(b, a, chunk_data, axis=0)
                
                # Spike detection on this chunk (adapted from spike_detection.py)
                med = np.median(np.absolute(filtered_chunk) / 0.6745, axis=0)
                
                for index in range(len(filtered_chunk)):
                    # Skip overlap region except for last chunk
                    if chunk_idx < num_chunks - 1 and index >= chunk_frames:
                        continue
                    
                    if points_pre < index < len(filtered_chunk) - points_post:
                        global_index = start_frame + index
                        
                        threshold_cross = filtered_chunk[index, :] < min_TH * med
                        threshold_arti = filtered_chunk[index, :] > max_TH * med
                        probable_spike = threshold_cross * threshold_arti
                        
                        if np.sum(probable_spike > 0):
                            for e in range(num_channels):
                                if probable_spike[e] == 1:
                                    channel_id = ids[e]
                                    channel_info = electrode_stream.channel_infos[channel_id]
                                    ch = int(channel_info.info['Label'][-2:])
                                    
                                    if ch not in reject_ch:
                                        t_diff = global_index - spike_times_per_channel[e][-1]
                                        if t_diff > refrec_period * fsample:
                                            # Check if local minimum
                                            if filtered_chunk[index, e] == np.min(
                                                    filtered_chunk[max(0, index-points_pre):min(len(filtered_chunk), index+points_post), e]):
                                                spike_times_per_channel[e].append(global_index)
                                                
                                                # Extract waveform if within bounds
                                                if global_index + points_post < total_frames and index >= points_pre:
                                                    spk_wave = list(filtered_chunk[index-points_pre:index+points_post, e])
                                                    spk_wave.insert(0, global_index)  # spike time
                                                    spk_wave.insert(0, ch)  # channel
                                                    all_spikes.append(spk_wave)
                
                # Free memory
                del chunk_data, filtered_chunk
                import gc
                gc.collect()
        
        print(f"[chunked] Total spikes detected: {len(all_spikes)}")
        
        # Save results
        if len(all_spikes) == 0:
            print("Warning: No spikes detected.")
            results = {"Filename": config.FILE_NAME, "Sampling rate": fsample, 
                       "Recording len": total_frames / fsample, "Raw_spikes": np.array([])}
        else:
            dat_arr = np.array(all_spikes)
            results = {"Filename": config.FILE_NAME, "Sampling rate": fsample,
                       "Recording len": total_frames / fsample, 
                       "Raw_spikes": dat_arr[dat_arr[:, 1].argsort()]}
        
        save_path = config.DATA_PATH.rpartition('/')[0]
        output_file = f"{save_path}/Spike_File_{config.FILE_NAME}.pkl"
        with open(output_file, 'wb+') as f:
            pickle.dump(results, f, -1)
        print(f"[chunked] Saved spikes to {output_file}")

    def prepare_data(self):
        print('---' * 30)
        print('PREPARING DATA...')
        dataset, dataset_test, self.pretraining_config, self.finetune_config = data_preparation(self.model_config,
                                                                                                self.data_preprocessing_config,
                                                                                                self.pretraining_config,
                                                                                                self.finetune_config,
                                                                                                self.benchmark)
        return dataset, dataset_test

    def initialize_model(self):
        model = model_initializer(self.model_config)
        return model

    def initialize_wandb(self, method):
        wandb_initializer(self.model_config, self.pretraining_config, self.finetune_config, method)

    def pretrain(self, model, dataset, dataset_test):
        print('---' * 30)
        print('PRETRAINING MODEL...')

        pretrain_model(model=model,
                       model_config=self.model_config,
                       pretraining_config=self.pretraining_config,
                       pretrain_method=self.pretrain_method,
                       dataset=dataset,
                       dataset_test=dataset_test,
                       save_weights=self.pretraining_config.SAVE_WEIGHTS,
                       save_dir=self.pretraining_config.SAVE_DIR)

    def finetune(self, model, dataset, dataset_test):
        print('---' * 30)
        print('FINETUNING MODEL...')
        y_finetuned, y_true = finetune_model(model=model, finetune_config=self.finetune_config,
                                     finetune_method=self.fine_tune_method,
                                     dataset=dataset, dataset_test=dataset_test, benchmark=self.benchmark)

        return y_finetuned, y_true

    def predict(self, model, dataset, dataset_test):
        encoded_data, encoded_data_test, y_true, y_true_test = model_predict_latents(model=model,
                                                                                     dataset=dataset,
                                                                                     dataset_test=dataset_test)

        return encoded_data, encoded_data_test, y_true, y_true_test

    def cluster_data(self, encoded_data, encoded_data_test):
        print('---' * 30)
        print('CLUSTERING...')
        y_pred, n_clusters = clustering(data=encoded_data,
                                        method=self.pretraining_config.CLUSTERING_METHOD,
                                        n_clusters=self.pretraining_config.N_CLUSTERS,
                                        eps=self.pretraining_config.EPS,
                                        min_cluster_size=self.pretraining_config.MIN_CLUSTER_SIZE,
                                        knn=self.pretraining_config.KNN)
        y_pred_test, n_clusters_test = clustering(data=encoded_data_test,
                                                  method=self.pretraining_config.CLUSTERING_METHOD,
                                                  n_clusters=self.pretraining_config.N_CLUSTERS,
                                                  eps=self.pretraining_config.EPS,
                                                  min_cluster_size=self.pretraining_config.MIN_CLUSTER_SIZE,
                                                  knn=self.pretraining_config.KNN)

        return y_pred, n_clusters, y_pred_test, n_clusters_test

    def evaluate_spike_sorting(self, y_pred, y_true, y_pred_test=None, y_true_test=None):
        print('---' * 30)
        print('EVALUATE RESULTS...')
        train_acc, test_acc = evaluate_clustering(y_pred, y_true, y_pred_test, y_true_test)
        return train_acc, test_acc

    def execute_pretrain(self):
        if self.benchmark:
            start_time = time.time()
            self.initialize_wandb(self.pretrain_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            self.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
            encoded_data, encoded_data_test, y_true, y_true_test = self.predict(model=model,
                                                                                dataset=dataset,
                                                                                dataset_test=dataset_test)
            y_pred, n_clusters, y_pred_test, n_clusters_test = self.cluster_data(encoded_data=encoded_data,
                                                                                 encoded_data_test=encoded_data_test)

            train_acc, test_acc = self.evaluate_spike_sorting(y_pred, y_true, y_pred_test, y_true_test)
            print("Train Accuracy: ", train_acc)
            print("Test Accuracy: ", test_acc)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)
        else:
            start_time = time.time()
            self.initialize_wandb(self.pretrain_method)
            self.extract_spikes_from_raw_recording()
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            self.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
            encoded_data, encoded_data_test, y_true, y_true_test = self.predict(model=model,
                                                                                dataset=dataset,
                                                                                dataset_test=dataset_test)
            y_pred, n_clusters, y_pred_test, n_clusters_test = self.cluster_data(encoded_data=encoded_data,
                                                                                 encoded_data_test=encoded_data_test)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)

    def execute_finetune(self):
        if self.benchmark:
            start_time = time.time()
            self.initialize_wandb(self.fine_tune_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            y_pred_finetuned, y_true = self.finetune(model=model, dataset=dataset, dataset_test=dataset_test)
            train_acc, _ = self.evaluate_spike_sorting(y_pred_finetuned, y_true)
            print("Accuracy after Finetuning: ", train_acc)
            end_time = time.time()
            print("Time Finetuning Execution: ", end_time - start_time)
        else:
            start_time = time.time()
            self.initialize_wandb(self.fine_tune_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            y_pred_finetuned, y_true = self.finetune(model=model, dataset=dataset, dataset_test=dataset_test)

            end_time = time.time()
            print("Time Finetuning Execution: ", end_time - start_time)

