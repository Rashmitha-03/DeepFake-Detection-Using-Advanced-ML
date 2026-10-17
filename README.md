## DeepFake Detection (DFDC) Solution by @selimsef

## Challenge details:

[Kaggle Challenge Page](https://www.kaggle.com/c/deepfake-detection-challenge)


### Fake detection articles  
- [The Deepfake Detection Challenge (DFDC) Preview Dataset](https://arxiv.org/abs/1910.08854)
- [Deep Fake Image Detection Based on Pairwise Learning](https://www.mdpi.com/2076-3417/10/1/370)
- [DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection](https://arxiv.org/abs/2001.03024)
- [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)

## Solution description 
In general solution is based on frame-by-frame classification approach. Other complex things did not work so well on public leaderboard.

#### Face-Detector
MTCNN detector is chosen due to kernel time limits. It would be better to use S3FD detector as more precise and robust, but opensource Pytorch implementations don't have a license. 

Input size for face detector was calculated for each video depending on video resolution.

- 2x scale for videos with less than 300 pixels wider side
- no rescale for videos with wider side between 300 and 1000
- 0.5x scale for videos with wider side > 1000 pixels
- 0.33x scale for videos with wider side > 1900 pixels

### Input size
As soon as I discovered that EfficientNets significantly outperform other encoders I used only them in my solution.
As I started with B4 I decided to use "native" size for that network (380x380).
Due to memory costraints I did not increase input size even for B7 encoder.

## Data preparation

Once DFDC dataset is downloaded all the scripts expect to have `dfdc_train_xxx` folders under data root directory. 

Preprocessing is done in a single script **`preprocess_data.sh`** which requires dataset directory as first argument. 
It will execute the steps below:  

##### 1. Find face bboxes
To extract face bboxes I used facenet library, basically only MTCNN. 
`python preprocessing/detect_original_faces.py --root-dir DATA_ROOT`
This script will detect faces in real videos and store them as jsons in DATA_ROOT/bboxes directory

##### 2. Extract crops from videos
To extract image crops I used bboxes saved before. It will use bounding boxes from original videos for face videos as well.
`python preprocessing/extract_crops.py --root-dir DATA_ROOT --crops-dir crops`
This script will extract face crops from videos and save them in DATA_ROOT/crops directory
 
##### 3. Generate landmarks
From the saved crops it is quite fast to process crops with MTCNN and extract landmarks  
`python preprocessing/generate_landmarks.py --root-dir DATA_ROOT`
This script will extract landmarks and save them in DATA_ROOT/landmarks directory
 
##### 4. Generate diff SSIM masks
`python preprocessing/generate_diffs.py --root-dir DATA_ROOT`
This script will extract SSIM difference masks between real and fake images and save them in DATA_ROOT/diffs directory

##### 5. Generate folds
`python preprocessing/generate_folds.py --root-dir DATA_ROOT --out folds.csv`
By default it will use 16 splits to have 0-2 folders as a holdout set. Though only 400 videos can be used for validation as well. 


## Training

Training 5 B7 models with different seeds is done in **`train.sh`** script.

During training checkpoints are saved for every epoch.

## Hardware requirements
Mostly trained on devbox configuration with 4xTitan V, thanks to Nvidia and DSB2018 competition where I got these gpus https://www.kaggle.com/c/data-science-bowl-2018/
 
Overall training requires 4 GPUs with 12gb+ memory. 
Batch size needs to be adjusted for standard 1080Ti or 2080Ti graphic cards.

As I computed fake loss and real loss separately inside each batch, results might be better with larger batch size, for example on V100 gpus. 
Even though SyncBN is used larger batch on each GPU will lead to less noise as DFDC dataset has some fakes where face detector failed and face crops are not really fakes.   





