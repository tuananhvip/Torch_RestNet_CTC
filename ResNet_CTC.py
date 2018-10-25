
# coding: utf-8

# # Install Pytorch
 #- http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'
print("platform {},cuda_output {},accelerator {} ".format(platform,cuda_output,accelerator) )

#Linux:
# <Lay link ban preview moi nhat cho Linux moi co CTC>: thay doi cu92 ==> cuda that: vd: cu90
!pip install pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
!pip install torchvision

#Windows: khong co CTC
# conda install pytorch -c pytorch
# pip3 install torchvision
import torch
print('CUDA:',torch.cuda.is_available())


# # Import

# In[66]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3'
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import time
print(time.asctime())


# # Main

# ## Load Database

# In[67]:


from os.path import join,exists
pHomeData='/home/u/AudioDBs/Aishell_compress/' 
prjNAME  ='Train_kws_renew_01' #  
pAPrjInfo=pHomeData+prjNAME

pchkPoint   = join(pAPrjInfo,'01_chkPoint.pkl')
plog_dir    = join(pAPrjInfo,'01_logs/')
pModel      = join(pAPrjInfo,'01_model.pkl')


pTrain   ='wavs/train/'
pTest    ='wavs/test/'
fnExt    ='.wav'

pWavs    =pHomeData+pTrain
pLabels  =pHomeData+'transcript/aishell_transcript_v0.8.txt'
pTestWavs=pHomeData+pTest


if not exists(pAPrjInfo):os.makedirs(pAPrjInfo)

#--------------------------------------------------------------------------------

pproject_info=join(pAPrjInfo,'project_info.json')

pAiPath_and_Labels       = join(pAPrjInfo,'AiPath_and_Labels.npy')
pAiPath_and_Labels_train = join(pAPrjInfo,'AiPath_and_Labels_train.npy')
pAiPath_and_Labels__test = join(pAPrjInfo,'AiPath_and_Labels__test.npy')

pAiMFCCs     = join(pAPrjInfo,'AiMFCCs.npy')
pAiMFCCs_test= join(pAPrjInfo,'AiMFCCs_test.npy')

pchar_vec   = join(pAPrjInfo,'char_vec.npy')
pchar_vec_test= join(pAPrjInfo,'SR_char_vec.npy')

pchar_len= join(pAPrjInfo,'char_length.npy')
pchar_len_test= join(pAPrjInfo,'SR_char_len.npy')

pAiLabels     = join(pAPrjInfo,'AiLabels.npy')
pAiLabels_test= join(pAPrjInfo,'SR_AiLabels.npy')

pLabels_kws = join(pAPrjInfo,'Labels_kws.npy')
pLabels_kws_test  = join(pAPrjInfo,'SR_Labels.npy')

def fnData_load(key_name):
    with open(pproject_info) as data_file:data_loaded = json.load(data_file)
    vl=data_loaded[key_name]
    return vl

########################################
pTrain_wav_longer__07_88=pHomeData+'Train_wav_longer__07_88.txt'
pTest_wav_longer__07_88 =pHomeData+'Test_wav_longer__07_88.txt'

char_vec=np.load(pchar_vec)
char_len=np.load(pchar_len)
maxlen_char =fnData_load('maxlen_char')

AllMFCCs     =np.load(pAiMFCCs)
Max_len_MFCC=fnData_load('Max_len_MFCC')
NumSamp=600
AllMFCCs=AllMFCCs[:NumSamp]
char_vec=char_vec[:NumSamp]
char_len=char_len[:NumSamp]
maxlen_mfcc_char=np.array([Max_len_MFCC,maxlen_char])
np.savez(join(pAPrjInfo,'AllMFCCs__char_vec__char_len__maxlen_mfcc_char.npz'), AllMFCCs,char_vec,char_len,maxlen_mfcc_char)


print('AllMFCCs:',AllMFCCs.shape,AllMFCCs.min(),AllMFCCs.max())

print('char_vec:',char_vec.shape,char_vec.min(),char_vec.max())#,char_vec[:3][:3])
print('char_len:',char_len.shape,char_len.min(),char_len.max())#,char_len[:2])

print('maxlen_char:',maxlen_char)
print('Max_len_MFCC=',Max_len_MFCC)    


BatchSize=32
inLen_MaxMFCC=AllMFCCs.shape[1]

Batch_Input=torch.Tensor(AllMFCCs[:BatchSize])

Batch_char_vec=torch.Tensor(char_vec[:BatchSize]).long()
Batch_char_len=torch.Tensor(char_len[:BatchSize]).long()

print('----------CONVERT -3D TO 4D-------NP ARRAY 2 TORCH------')
Batch_Input=AllMFCCs[:32]
print('bat inp:',Batch_Input.shape)
def ConvertNpArray3D_2Tensor4D(NpArray3D):
    bat=[]
    for k,mfcc in enumerate(NpArray3D):
        bat.append([mfcc])
    bat=torch.Tensor(bat)
    return bat
bat=ConvertNpArray3D_2Tensor4D(Batch_Input)
print('bat out:',bat.shape)
print('---------------------------------------------------------')
print('Batch_Input:',Batch_Input.shape)
print('Batch_char_vec:',Batch_char_vec.shape)
print('Batch_char_len:',Batch_char_len.shape)


# In[68]:


from os.path import join,exists
pHomeData='/home/u/AudioDBs/Aishell_compress/' 
prjNAME  ='Train_kws_renew_01'    
pAPrjInfo=pHomeData+prjNAME

pData=join(pAPrjInfo,'AllMFCCs__char_vec__char_len__maxlen_mfcc_char.npz')
if not exists(pData):
    pData='AllMFCCs__char_vec__char_len__maxlen_mfcc_char.npz'

data_files=np.load(pData)
print(data_files.files)

AllMFCCs=data_files['arr_0']
char_vec=data_files['arr_1']
char_len=data_files['arr_2']
Max_len_MFCC=data_files['arr_3'][0]
maxlen_char =data_files['arr_3'][1]
print(AllMFCCs.shape)
print(char_vec.shape)
print(char_len.shape)
print(Max_len_MFCC)
print(maxlen_char)
OneBatch=AllMFCCs[:32]
print(OneBatch.shape)


# # Train CTC

# ## Model

# In[69]:


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=256*8):
        super(ResNet, self).__init__()
        Nsize=32
        self.in_planes = Nsize
        self.conv1 = nn.Conv2d(1, Nsize, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Nsize)
        self.layer1 = self._make_layer(block,Nsize,num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(3840, num_classes) #
        self.Smax   = nn.LogSoftmax(dim=1)
#         self.Smax   = nn.Softmax(dim=-1)
        
    def _make_layer(self, block, planes, num_blocks, stride):
#         print('_make_layer:',block, planes, num_blocks, stride)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def Change_dimention(self,inp,dim=(-1)):
        MFs=[]
        for mf in inp:
            mf=mf.view(dim)
            MFs.append(mf)
        MFs = torch.stack(MFs)
        return MFs
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)));        #print(1,out.size())
        out = self.layer1(out);         #print(2,out.size())
        out = self.layer2(out);         #print(3,out.size())
        out = self.layer3(out);         #print(4,out.size())
        out = self.layer4(out);         #print(5,out.size())
        out = F.avg_pool2d(out, 4);     #print(6,out.size())
#       out = out.view(out.size(0), -1);#print(7,out.size())
        
        out=self.Change_dimention(out,(-1))
        
        out = self.linear(out);         #print(8,out.size())

        out=self.Change_dimention(out,(-1,8))

        out = self.Smax(out);      
        return out

def ResNet18():    return ResNet(BasicBlock, [2,2,2,2])

OneBatch=Batch_Input.reshape(OneBatch.shape[0], 1, 247, 20) # [# 32, 247, 20] ==> [32, 1, 247, 20]
OneBatch=torch.Tensor(OneBatch)

def test():
    net = ResNet18()
    
    y = net(OneBatch) #(32, 247, 20)
    print(y.size())
#     print(y)
#     print(net)

test()
import time;print(time.asctime())


# ## Train

# In[70]:


from torch import nn 
from tensorflow.python.ops import array_ops
from torch import nn, autograd, FloatTensor, optim

ctc_loss       = nn.CTCLoss(reduction='elementwise_mean')
net = ResNet18()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate=0.01
optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print(time.asctime())

net.train()
train_loss = 0
correct = 0
total = 0

k=0
BatchSize=32
print('Net out shape: [batch,256,8]\n self.Smax   = nn.LogSoftmax(dim=1)')
for _ in range(1):
    for batch_idx in range(0, len(AllMFCCs),BatchSize):
      # Get data:
        Batch_Input   = AllMFCCs[batch_idx:BatchSize+batch_idx]
        target_lengths= char_len[batch_idx:BatchSize+batch_idx]
        targets       = char_vec[batch_idx:BatchSize+batch_idx]
        targets       = targets+1
        targets       =torch.Tensor(targets).long()
        target_lengths=torch.Tensor(target_lengths).long()

     # Convert to correct dimentions:
        Batch_Input1=ConvertNpArray3D_2Tensor4D(Batch_Input) #Batch_Input1=Batch_Input.reshape(Batch_Input.shape[0], 1, 247, 20)
        # bat inp: (32, 247, 20)
        # bat out: torch.Size([32, 1, 247, 20]) 

     # put in to GPU:
        Batch_Input1=autograd.Variable(torch.Tensor(Batch_Input1))
        targets=autograd.Variable(targets)

        Batch_Input1,targets = Batch_Input1.to(device), targets.to(device)

        log_probs=net(Batch_Input1)

     # Prepair to CTC input:    
        log_probs    = log_probs.transpose(1,0) 
        input_lengths= torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.long); 
        input_lengths = autograd.Variable(input_lengths)
        target_lengths= autograd.Variable(target_lengths)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
     #Update Weight:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        print('Batch:{} - Loss:{:>20} - Total Loss:{:>20}'.format(k,loss.item(),train_loss))
        k+=1
#     if k==100: break
    
#     _, predicted = log_probs.max(1)
#     total += targets.size(0)
#     correct += predicted.eq(targets).sum().item()


#     print('\rTrain:',batch_idx,'/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total), end='   ',flush=True)
print('Done!')
 


# In[71]:


log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 21, (16, 30), dtype=torch.long)
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()
loss


# ## Train-End

# Signature:   ctc_loss(*input, **kwargs)
# Type:        CTCLoss
# String form: CTCLoss()
# File:        ~/anaconda3/envs/tf/lib/python3.6/site-packages/torch/nn/modules/loss.py
# Docstring:  
# The Connectionist Temporal Classification loss.
# 
# Args:
#     blank (int, optional): blank label. Default :math:`0`.
#     reduction (string, optional): Specifies the reduction to apply to the output:
#         'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
#         'elementwise_mean': the output losses will be divided by the target lengths and
#         then the mean over the batch is taken. Default: 'elementwise_mean'
# 
# Inputs:
# 
#     log_probs: Tensor of size :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
#         `T = input length`, and `N = batch size`.
#         The logarithmized probabilities of the outputs
#         (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
#         
#     targets: Tensor of size :math:`(N, S)` or `(sum(target_lenghts))`.
#         Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
#         
#     input_lengths: Tuple or tensor of size :math:`(N)`.
#         Lengths of the inputs (must each be :math:`\leq T`)
#         
#     target_lengths: Tuple or tensor of size  :math:`(N)`.
#         Lengths of the targets
# 
# 
# Example::
# 
#     >>> ctc_loss = nn.CTCLoss()
#     >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
#     >>> targets = torch.randint(1, 21, (16, 30), dtype=torch.long)
#     >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
#     >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
#     >>> loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
#     >>> loss.backward()
# 
# Reference:
#     A. Graves et al.: Connectionist Temporal Classification:
#     Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
#     https://www.cs.toronto.edu/~graves/icml_2006.pdf
# 
# .. Note::
#     In order to use CuDNN, the following must be satisfied: :attr:`targets` must be
#     in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
#     :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
#     dtype :attr:`torch.int32`.
# 
#     The regular implementation uses the (more common in PyTorch) `torch.long` dtype.
# 
# 
# .. include:: cudnn_deterministic.rst
 Args:
    labels: An `int32` `SparseTensor`.
      `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores the id for (batch b, time t).
      `labels.values[i]` must take on values in `[0, num_labels)`.
      See `core/ops/ctc_ops.cc` for more details.
      
    inputs: 3-D `float` `Tensor`.
      If time_major == False, this will be a `Tensor` shaped:
        `[batch_size, max_time, num_classes]`.
      If time_major == True (default), this will be a `Tensor` shaped:
        `[max_time, batch_size, num_classes]`.
      The logits.
      
    sequence_length: 1-D `int32` vector, size `[batch_size]`.
      The sequence lengths.
      
    preprocess_collapse_repeated: Boolean.  Default: False.
      If True, repeated labels are collapsed prior to the CTC calculation.
    ctc_merge_repeated: Boolean.  Default: True.
    ignore_longer_outputs_than_inputs: Boolean. Default: False.
      If True, sequences with longer outputs than inputs will be ignored.
      
    time_major: The shape format of the `inputs` Tensors.
      If True, these `Tensors` must be shaped `[max_time, batch_size,      num_classes]`.
      If False, these `Tensors` must be shaped `[batch_size, max_time,      num_classes]`.
      Using `time_major = True` (default) is a bit more efficient because it
      avoids transposes at the beginning of the ctc_loss calculation.  However,
      most TensorFlow data is batch-major, so by this function also accepts
      inputs in batch-major form.

  Returns:
    A 1-D `float` `Tensor`, size `[batch]`, containing the negative log
      probabilities.