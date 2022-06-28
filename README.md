# WeakCounterF

## Directory Overview
The source codes and datasets we used to write our paper are located in the "code" and "data" directories, respectively.<br>
The "requirement.txt" file describes the Python modules of the experimental environment used in this paper. Please use it as a reference for constructing the environment when trying out the experiments.

## "data" directory
There are several directories within the data directory.<br>
The "raw_data" folder contains raw data for use in pre-training and testing.<br>
The folder "few-shot_data" contains raw data for fine tuning.<br>
The other folders are empty, but each program for Data Augmentation, Label Diversification, and extraction of a single movement will contain the corresponding data in the folders.
- The "augmented_data" folder contains data that has undergone Data Augmentation.
- The "double", "triple", and "plus" folders contain data that has undergone Label Diversification.
- The "one_timeaction_opt" folder stores the data extracted for one operation from the data in "few-shot_data".
- The "pre-trained_model" folder contains model weight files such as pre-trained models.

## "code" directory
The code directory contains all the programs necessary for the experiments conducted in this paper, but with modifications for submission of supplementary materials. <br>
It would be appreciated if you could modify the paths in data loading, etc. as appropriate.<br>
Details of each program are as follows.

- The "keras_utils.py" and "pre_utils.py" files are modules for smooth execution of each program.
- The "Data Augmentation.ipynb" file is used to perform data augmentation on "raw_data" and store the augmented data in the "augmented_data" folder.
- The "Label Diversification_(Double,Triple,Plus).ipynb" file is a label diversification file for "raw_data". It also contains code to perform data expansion after label diversification.
- The "Extract_1time_Action.ipynb" is a program to extract one-time action data from "few-shot_data", that is, data containing a small number of actions, and store it in the "one_time_action_opt" folder.
- The "Pre-Train(WeakCounter).ipynb" is a program for pre-training against WeakCounter-Net with the LeaveOne user-out. This pragram is equivalent to the comparison method WeakCounter.
- The "Fine-Tuning(WeakCounterF_one(few)-shot.ipynb)" is a program to fine-tune a pre-trained model by creating synthetic data from a single extracted movement.
As the file name indicates, one-shot and few-shot are separated in the file.
- The "CNN(compare method).ipynb" and the "Only-composite.ipynb" are programs that correspond to the "CNN" and "Only-composite" comparison methods, respectively.

## Procedures
1. Data Augmentation and Label Diversification (3 types) for all "raw_data" of all subjects.(Data_Augmentation.ipynb,Label_Diversification_Double.ipynb,Label_Diversification_Triple.ipynb,Label_Diversification_Plus.ipynb)
2. Perform a single motion extraction for all "few-shot_data" of all motions for all subjects.(Extract_1time_Action.ipynb)
3. Pre-training with data from users other than the target user.(Pre-Train(WeakCounter).ipynb)
4. Fine tuning with composed data.(Fine-Tuning(WeakCounteF_one-shot).ipynb or Fine-Tuning(WeakCounteF_few-shot).ipynb)
5. Experiments will also be conducted and compared for comparison methods.(CNN(compare method).ipynb,Only-composite.ipynb)


## Contact information
If you have any questions about the source code or dataset, please contact.<br>

Yuuki Nishino: nishino.yuuki@ist.osaka-u.ac.jp <br>
Takuya Maekawa: maekawa@ist.osaka-u.ac.jp
