import os
import numpy as np
import io
from PIL import Image

import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='',clip_len = 16):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        # self.name = "{:s}_{:s}_{:s}_{:.0f}%-{:d}".format(dataset, rule, str(rule_arg), args.active_ratio*args.total_client, args.total_client)
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.clip_len = clip_len
        self.set_data()

    def _is_ucf101_ma(self):
        return self.dataset in ('UCF101-MA', 'ucf101_ma', 'ucf101-ma')

    def _resolve_ucf101_ma_root(self):
        candidate_roots = [
            '/home/omnisky/FL-main/data/UCF101-MA',
            os.path.join(self.data_path, 'data', 'UCF101-MA'),
            os.path.join(self.data_path, 'Data', 'Raw', 'UCF101-MA'),
            os.path.join(os.path.abspath(self.data_path if self.data_path else '.'), 'data', 'UCF101-MA'),
            os.path.join(os.path.abspath(self.data_path if self.data_path else '.'), 'Data', 'Raw', 'UCF101-MA'),
        ]
        for root in candidate_roots:
            if os.path.isdir(root):
                return root
        raise FileNotFoundError(
            "UCF101-MA dataset not found. Tried: {}".format(', '.join(candidate_roots))
        )

    def _extract_video_middle_frame(self, video_path, image_size=(32, 32)):
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        success, frame = capture.read()
        if (not success or frame is None) and frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = capture.read()

        capture.release()
        if not success or frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        return np.transpose(frame, (2, 0, 1))

    def _build_ucf101_ma_arrays(self, image_size=(32, 32), test_size=0.2):
        root_dir = self._resolve_ucf101_ma_root()
        class_names = [
            folder for folder in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, folder))
        ]
        if len(class_names) == 0:
            raise RuntimeError("No class folders found in {}".format(root_dir))

        label_to_index = {name: idx for idx, name in enumerate(class_names)}
        frames = []
        labels = []
        failed_files = []

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            video_files = sorted(
                [file for file in os.listdir(class_dir) if file.lower().endswith('.avi')]
            )
            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                frame = self._extract_video_middle_frame(video_path, image_size=image_size)
                if frame is None:
                    failed_files.append(video_path)
                    continue
                frames.append(frame)
                labels.append(label_to_index[class_name])

        if len(frames) == 0:
            raise RuntimeError("No valid videos were decoded from {}".format(root_dir))

        data_x = np.asarray(frames, dtype=np.float32)
        data_y = np.asarray(labels, dtype=np.int64)

        stratify_labels = data_y if len(np.unique(data_y)) > 1 else None
        try:
            train_x, test_x, train_y, test_y = train_test_split(
                data_x,
                data_y,
                test_size=test_size,
                random_state=self.seed,
                stratify=stratify_labels,
                shuffle=True
            )
        except ValueError:
            train_x, test_x, train_y, test_y = train_test_split(
                data_x,
                data_y,
                test_size=test_size,
                random_state=self.seed,
                shuffle=True
            )

        return train_x, train_y.reshape(-1, 1), test_x, test_y.reshape(-1, 1), label_to_index, failed_files

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            # Get Raw data
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trainset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                testset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trainset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trainset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            
            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([# transforms.Resize(224),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], #pre-train
                                                #                      std=[0.229, 0.224, 0.225])])
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                     std=[0.5, 0.5, 0.5])])
                # trainset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=True , download=True, transform=transform)
                # testset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=False, download=True, transform=transform)
                # root_dir = self.data_path
                root_dir = "./Data/Raw/tiny-imagenet-200/"
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')
                with open(trn_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))
                with open(tst_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))
                trainset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list, transformer=transform)
                testset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list, transformer=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;

                # face classification
            if self.dataset == 'face_dataset':
                print(self.dataset)
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                        # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                root_dir = os.path.join(self.data_path, 'Data/Raw/face_dataset/')
                trainset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'train'),
                                                                transform=transform)
                testset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'test'), transform=transform)
                for i, (img, label) in enumerate(trainset):
                    print(f"Sample {i} path: {trainset.samples[i][0]}, label: {label}")
                    if i >= 2: break
                train_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False,
                                                             num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False,
                                                            num_workers=0)
                self.channels = 3
                self.width = 64
                self.height = 64
                self.n_cls = 8
                print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")
                #print(f"X_data shape: {self.train_x.shape}, y_data shape: {self.train_y.shape}")
                #print(f"X_data dtype: {self.train_x.dtype}, y_data dtype: {self.train_y.dtype}")

            #将视频数据集处理为图像集
            if self.dataset in('ucf101','hmdb51'):
                print(f"Processing video dataset: {self.dataset}")

                if self.dataset == 'ucf101':
                    root_dir = os.path.join(self.data_path, 'Data/Raw/ucf101/')
                    save_dir = os.path.join(self.data_path,'Data/Raw/ucf101_jpg/')
                else:
                    root_dir = os.path.join(self.data_path, 'Data/Raw/hmdb51_org/')
                    save_dir = os.path.join(self.data_path,'Data/Raw/hmdb51_jpg/')

                for file in os.listdir(root_dir):
                    file_path = os.path.join(root_dir, file)
                    video_files = [name for name in os.listdir(file_path)]
                    train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                    # 再从训练集中分出验证集
                    train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                    train_dir = os.path.join(save_dir, 'train', file)
                    val_dir = os.path.join(save_dir, 'val', file)
                    test_dir = os.path.join(save_dir, 'test', file)

                    if not os.path.exists(train_dir):
                        os.makedirs(train_dir,exist_ok=True)
                    if not os.path.exists(val_dir):
                        os.makedirs(val_dir,exist_ok=True)
                    if not os.path.exists(test_dir):
                        os.makedirs(test_dir,exist_ok=True)
                    for video in train:
                        self.process_video(video, file, train_dir)
                    for video in val:
                        self.process_video(video, file, val_dir)
                    for video in test:
                        self.process_video(video, file, test_dir)

                trainset = Dataset(dataset_name='hmdb51', split='train', clip_len=self.clip_len)
                testset = Dataset(dataset_name='hmdb51', split='test', clip_len=self.clip_len)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False,
                                                         num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False,
                                                        num_workers=0)
                self.channels = 3
                self.width = 171
                self.height = 128
                self.n_cls = 50
                print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")

            if self._is_ucf101_ma():
                print("Processing video dataset: UCF101-MA")
                train_x, train_y, test_x, test_y, label_to_index, failed_files = self._build_ucf101_ma_arrays(
                    image_size=(32, 32), test_size=0.2
                )
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = len(label_to_index)
                if len(failed_files) > 0:
                    print("Warning: {} videos failed to decode and were skipped.".format(len(failed_files)))
                print("UCF101-MA classes: {}, train videos: {}, test videos: {}".format(
                    self.n_cls, train_x.shape[0], test_x.shape[0]
                ))
            
            if self.dataset != 'emnist' and not self._is_ucf101_ma():
                train_itr = train_load.__iter__(); test_itr = test_load.__iter__() 
                # labels are of shape (n_data,)
                train_x, train_y = train_itr.__next__()
                test_x, test_y = test_itr.__next__()

                train_x = train_x.numpy(); train_y = train_y.numpy().reshape(-1,1)
                test_x = test_x.numpy(); test_y = test_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                train_idx = np.where(y_train < 10)[0]

                y_train = y_train[train_idx]
                x_train = x_train[train_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                test_idx = np.where(y_test < 10)[0]

                y_test = y_test[test_idx]
                x_test = x_test[test_idx]
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                train_x = (x_train - mean_x) / std_x
                train_y = y_train
                
                test_x = (x_test  - mean_x) / std_x
                test_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]
            
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
            
            
            ###
            n_data_per_client = int((len(train_y)) / self.n_client)
            # Draw from lognormal distribution
            # client_data_list = (np.random.lognormal(mean=np.log(n_data_per_client), sigma=self.unbalanced_sgm, size=self.n_client))
            # client_data_list = (client_data_list/(np.sum(client_data_list)/len(train_y)))
            client_data_list = np.ones(self.n_client, dtype=int)*n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break
            ###     
            
            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                    # np.save("results/heterogeneity_distribution_{:s}.npy".format(self.dataset), cls_priors)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client,self.n_cls])
                    a[:,c::] = 0
                    [np.random.shuffle(i) for i in a]
                    # np.save("results/heterogeneity_distribution_{:s}_{:s}.npy".format(self.dataset, self.rule), a/c)
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i,j] != 0:
                                prior_cumsum[i,j] = a[i,0:j+1].sum()/c*1.0

                idx_list = [np.where(train_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                # print(cls_amount)
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
    
                while(np.sum(client_data_list)!=0):
                    curr_client = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    # print('Remaining Data: %d' %np.sum(client_data_list))
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            cls_amount [cls_label] = len(idx_list[cls_label]) 
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)
                
                # cls_means = np.zeros((self.n_client, self.n_cls))
                # for client in range(self.n_client):
                #     for cls in range(self.n_cls):
                #         cls_means[client,cls] = np.mean(client_y[client]==cls)
                # prior_real_diff = np.abs(cls_means-cls_priors)
                # print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                # print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(train_y)//100 % self.n_client == 0 
                
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(train_y[:, 0])
                n_data_per_client = len(train_y) // self.n_client
                # client_x dtype needs to be float32, the same as weights
                client_x = np.zeros((self.n_client, n_data_per_client, 3, 32, 32), dtype=np.float32)
                client_y = np.zeros((self.n_client, n_data_per_client, 1), dtype=np.float32)
                train_x = train_x[idx] # 50000*3*32*32
                train_y = train_y[idx]
                n_cls_sample_per_device = n_data_per_client // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        client_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = train_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        client_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = train_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :]

            # 视频数据分配部分
            elif self.rule == 'iid' and self.dataset in ['ucf101', 'hmdb51']:
                # 以视频为单位做 IID 分配
                video_paths = np.array(trainset.fnames)
                video_labels = np.array(trainset.label_array)

                clip_len = trainset.clip_len
                crop_size = trainset.crop_size
                resize_height = trainset.resize_height
                resize_width = trainset.resize_width

                n_videos = len(video_paths)
                video_indices = np.random.permutation(n_videos)
                videos_per_client = n_videos // self.n_client

                client_x = []
                client_y = []

                for client_idx in range(self.n_client):
                    start_idx = client_idx * videos_per_client
                    end_idx = (client_idx + 1) * videos_per_client
                    client_video_indices = video_indices[start_idx:end_idx]

                    # 当前 client 的视频路径和标签
                    client_video_paths = video_paths[client_video_indices]
                    client_video_labels = video_labels[client_video_indices]

                    client_videos = []
                    for path in client_video_paths:
                        frame_buffer = trainset.load_frames(path)
                        cropped_frames = trainset.crop(frame_buffer,clip_len,crop_size)
                        normalized_frames = trainset.normalize(cropped_frames)
                        video_array = trainset.to_tensor(normalized_frames)
                        client_videos.append(video_array)

                    client_videos = np.array(client_videos, dtype=np.float32)

                    # 按视频保存（保持视频的完整性）
                    client_x.append(client_videos)  # 保存的是视频文件路径数组
                    client_y.append(client_video_labels.astype(np.int64))  # 保存的是对应的视频标签数组

                client_x = np.asarray(client_x, dtype=object)  # 每个元素是一个视频列表
                client_y = np.asarray(client_y, dtype=object)  # 每个元素是对应标签


            elif self.rule == 'iid':
                
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
            
                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                    client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)

            
            self.client_x = client_x; self.client_y = client_y

            self.test_x  = test_x;  self.test_y  = test_y
            
            # Save data
            print('begin to save data...')
            os.makedirs('%sData' % self.data_path, exist_ok=True)
            os.makedirs('%sData/%s' % (self.data_path, self.name), exist_ok=True)
            
            np.save('%sData/%s/client_x.npy' %(self.data_path, self.name), client_x)
            np.save('%sData/%s/client_y.npy' %(self.data_path, self.name), client_y)

            np.save('%sData/%s/test_x.npy'  %(self.data_path, self.name),  test_x)
            np.save('%sData/%s/test_y.npy'  %(self.data_path, self.name),  test_y)

            print('data loading finished.')

        else:
            print("Data is already downloaded")
            self.client_x = np.load('%sData/%s/client_x.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.client_y = np.load('%sData/%s/client_y.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.n_client = len(self.client_x)

            self.test_x  = np.load('%sData/%s/test_x.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            self.test_y  = np.load('%sData/%s/test_y.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'tinyimagenet':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            #face_classification
            if self.dataset == 'face_dataset':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 8;
            if self.dataset == 'hmdb51':
                self.channels = 3; self.width = 171; self.height = 128; self.n_cls = 50;
            if self._is_ucf101_ma():
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = int(np.max(self.test_y)) + 1;
            
            print('data loading finished.')
                
        '''
        print('Class frequencies:')
        count = 0
        for client in range(self.n_client):
            print("Client %3d: " %client + 
                  ', '.join(["%.3f" %np.mean(self.client_y[client]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.client_y[client].shape[0])
            count += self.client_y[client].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.test_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.test_y.shape[0])
        '''
    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))
        # join路径把视频读进来（open cv读的）
        capture = cv2.VideoCapture(os.path.join('/Users/bubble/PycharmProjects/FL-Simulator/Data/Raw/hmdb51_org', action_name, video))

        # 帧数、宽度、长度
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        # 间隔多少帧取一下（相邻帧太过类似的话就没必要）
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            # 如果帧数小于16，那么间隔的帧数再减个1
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True
        # 依次读帧，然后间隔几帧保存成图片
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                # 先对数据进行resize操作
                if (frame_height != 128) or (frame_width != 171):
                    frame = cv2.resize(frame, (171, 128))
                # 尾缀用 .jpg
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # 把视频由一个视频的形式转为图像的形式
        # Release the VideoCapture once it is no longer needed
        capture.release()
        
def generate_syn_logistic(dimension, n_client, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points
    
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)
    
    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_client)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))
    
    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_client))
    data_y = list(range(n_client))

    mean_W = np.random.normal(0, alpha, n_client)
    B = np.random.normal(0, beta, n_client)

    mean_x = np.zeros((n_client, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_client):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))
    
    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))
    
    for i in range(n_client):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y

    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x = None, data_y=True, train=False, dataset_name='',split='train',clip_len = 16):
        self.name = dataset_name

        if self.name not in ('ucf101', 'hmdb51') and data_x is None:
            raise TypeError("Dataset.__init__() missing required argument 'data_x' for dataset '{}'".format(self.name))

        #视频数据集初始化
        elif self.name == 'ucf101' or self.name == 'hmdb51':
            self.root_dir = '/Users/bubble/PycharmProjects/FL-Simulator/Data/Raw/hmdb51_org'
            self.save_dir = '/Users/bubble/PycharmProjects/FL-Simulator/Data/Raw/hmdb51_jpg'

            folder = os.path.join(self.save_dir, split)
            self.clip_len = clip_len  # 样本时间序列的长度
            self.split = split  # train还是 val

            # 预处理当中指定的参数
            # The following three parameters are chosen as described in the paper section 4.1
            self.resize_height = 128
            self.resize_width = 171
            self.crop_size = 112  # resize完了之后再进行裁减


            # Obtain all the filenames of files inside all the class folders
            # Going through each class folder one at a time
            self.fnames, labels = [], []
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, label)):
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

            assert len(labels) == len(self.fnames)
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

            # Prepare a mapping between the label names (strings) and indices (ints)
            self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
            # Convert the list of label names into an array of label indices
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

            # if dataset_name == "ucf101":
            #     if not os.path.exists('dataloaders/ucf_labels.txt'):
            #         with open('dataloaders/ucf_labels.txt', 'w') as f:
            #             for id, label in enumerate(sorted(self.label2index)):
            #                 f.writelines(str(id + 1) + ' ' + label + '\n')
            #
            # elif dataset_name == 'hmdb51':
            #     if not os.path.exists('dataloaders/hmdb_labels.txt'):
            #         with open('dataloaders/hmdb_labels.txt', 'w') as f:
            #             for id, label in enumerate(sorted(self.label2index)):
            #                 f.writelines(str(id + 1) + ' ' + label + '\n')

        elif self.name == 'mnist' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet" or self.name == "face_dataset" or self.name in ('UCF101-MA', 'ucf101_ma', 'ucf101-ma'):
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
                
        else:
            raise NotImplementedError
            
           
    def __len__(self):
        if self.name == 'ucf101' or self.name == 'hmdb51':
            return len(self.fnames)
        else:
            return len(self.X_data)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # 第一步，先把所有的路径读取到
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        # 对数据进行读取（即便超过片段值，这里也是全部读取）
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame  # 把数据放入到指定的buffer当中

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        '''
        - 进行crop操作， 如果设定帧数是16，但是文件夹里有超过16的图片，我们需要用窗口形式截取固定帧数
        - crop同时还会在图片上裁减，比如224的图上crop出112的目标区域
        '''
        #

        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def __getitem__(self, idx):
        if self.name == 'ucf101' or self.name == 'hmdb51':
            # Loading and preprocessing.
            # 一共有8460个文件夹， index指明从哪个文件夹读取（buffer拿到的是一整个目录的图片的数据）
            buffer = self.load_frames(self.fnames[idx])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            # 拿到这个buffer里数据对应的标签
            labels = np.array(self.label_array[idx])

            if self.split == 'test':
                # Perform data augmentation
                buffer = self.randomflip(buffer)
            # 转换成tensor(如果训练的时和做了这些，那测试时也要进行normalize to_tensor)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            return torch.from_numpy(buffer), torch.from_numpy(labels)

        if self.name == 'mnist' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name in ('UCF101-MA', 'ucf101_ma', 'ucf101-ma'):
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
                
        elif self.name == 'tinyimagenet' or self.name == 'face_dataset':
            img = self.X_data[idx]
            #print(f"Raw img shape: {img.shape}, dtype: {img.dtype}")
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if np.random.rand() > .5:
                    # Random cropping
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            #print(f"Processed img shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                #print(f"Label: {y}")
                return img, y
        else:
            raise NotImplementedError
            
class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)



'''def set_data(self):
   if self.dataset == 'CIFAR10':
       # ... (CIFAR10 逻辑)
    elif self.dataset == 'tinyimagenet':
        # ... (Tiny ImageNet 逻辑)
    elif self.dataset == 'face_dataset':
        print(self.dataset)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        root_dir = os.path.join(self.data_path, 'Data/Raw/face_dataset/')
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'train'), transform=transform)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'test'), transform=transform)
        train_load = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False, num_workers=2)
        test_load = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
        self.channels = 3
        self.width = 224
        self.height = 224
        self.n_cls = len(trainset.classes)
    # ... (后续分配逻辑)'''
