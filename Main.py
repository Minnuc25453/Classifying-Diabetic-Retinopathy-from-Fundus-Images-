import os
import cv2 as cv
from numpy import matlib
from sklearn.utils import shuffle
import random as rn
from COA import COA
from DGO import DGO
from DOA import DOA
from Glob_Vars import Glob_Vars
from Model_AES import Model_AES
from Model_CNN import Model_CNN
from Model_DBN import Model_DBN
from Model_DES import Model_DES
from Model_ECC import Model_ECC
from Model_FLNN import Model_FLNN
from Model_IR_CNN import Model_IR_CNN
from Model_MWFF_CNN import Model_MWFF_CNN
from Model_MobileNet import Model_MobileNet
from Model_PWLCM import Model_PLCM
from Model_VGG16 import Model_VGG16
from Objective_Function import Objfun_Cls, Objective_Crypto
from PROPOSED import PROPOSED
from VIT_FEAT import VIT
from WPA import WPA
from Plot_Results import *
from Plot_pred import *


# Read Dataset
an = 0
if an == 1:
    path = './Dataset/colored_images'
    out_dir = os.listdir(path)
    Images = []
    Target = []
    for k in range(len(out_dir)):
        folder_path = path + '/' + out_dir[k]
        in_dir = os.listdir(folder_path)
        for j in range(len(in_dir)):
            print(k, j)
            Img_path = folder_path + '/' + in_dir[j]
            Image = cv.imread(Img_path)
            img_re = cv.resize(Image, [512, 512])
            Images.append(img_re)
            Target.append(out_dir[k])
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    Images, tar = shuffle(Images, tar)
    np.save('Image.npy', Images)
    np.save('Target.npy', tar)

# Feature Extraction for
# R- VIT1
an = 0
if an == 1:
    feat = []
    Images = np.load('Image.npy', allow_pickle=True)
    for j in range(len(Images)):
        print(j)
        img = Images[j]   # cv.cvtColor(Images[j], cv.COLOR_GRAY2BGR)
        Feat_1 = VIT(img)
        feat.append(Feat_1)
    np.save('FEAT_1.npy', feat)

# VGG16
an = 0
if an == 1:
    Images = np.load('Image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Feat_2 = Model_VGG16(Images, Target)
    np.save('Feat_2.npy', Feat_2)

# MobileNet
an = 0
if an == 1:
    Images = np.load('Image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Feat_3 = Model_MobileNet(Images, Target)
    np.save('Feat_3.npy', Feat_3)


# Optimization for Weighted Fused Feature
an = 0
if an == 1:
    Feat_1 = np.load('Feat_1.npy', allow_pickle=True)
    Feat_2 = np.load('Feat_2.npy', allow_pickle=True)
    Feat_3 = np.load('Feat_3.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Glob_Vars.Feat1 = Feat_1
    Glob_Vars.Feat2 = Feat_2
    Glob_Vars.Feat3 = Feat_3
    Glob_Vars.Target = Target
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat(([0.01]), Npop, Chlen)
    xmax = matlib.repmat(([0.09]), Npop, Chlen)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objfun_Cls
    max_iter = 50

    print('DOA....')
    [bestfit1, fitness1, bestsol1, Time1] = DOA(initsol, fname, xmin, xmax, max_iter)

    print('COA....')
    [bestfit2, fitness2, bestsol2, Time2] = COA(initsol, fname, xmin, xmax, max_iter)

    print('DGO....')
    [bestfit3, fitness3, bestsol3, Time3] = DGO(initsol, fname, xmin, xmax, max_iter)

    print('WPA....')
    [bestfit4, fitness4, bestsol4, Time4] = WPA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    fitness = [fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()]

    np.save('Bestsol.npy', sol)
    np.save('Fitness.npy', fitness)

# Optimized Weighted Feature Fusion
an = 0
if an == 1:
    Feat_1 = np.load('Feat_1.npy', allow_pickle=True)
    Feat_2 = np.load('Feat_2.npy', allow_pickle=True)
    Feat_3 = np.load('Feat_3.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    weight = sol[4, :][0]  # For Proposed Only
    Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=1)
    Weighted_Feature_Fusion = Feat * weight
    np.save('Weighted_Fused_Feature.npy', Weighted_Feature_Fusion)

# Classification
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Data = np.load('Weighted_Fused_Feature.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Steps_per_Epochs = ['4', '8', '16', '32', '48']
    Eval = []
    for m in range(len(Steps_per_Epochs)):
        per = round(Image.shape[0] * 0.75)
        EVAL = np.zeros((5, 25))
        train_data = Image[:per, :]
        train_target = Target[:per, :]
        test_data = Image[per:, :]
        test_target = Target[per:, :]
        EVAL[0, :] = Model_IR_CNN(Image, Target)
        EVAL[1, :] = Model_FLNN(train_data, train_target, test_data, test_target)
        EVAL[2, :] = Model_DBN(train_data, train_target, test_data, test_target)
        EVAL[3, :] = Model_CNN(train_data, train_target, test_data, test_target)
        EVAL[4, :] = Model_MWFF_CNN(Data, Target)
        Eval.append(EVAL)
    np.save('Eval_ALL.npy', Eval)

# Optimization for Cryptography
an = 0
if an == 1:
    Data = np.load('Image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Glob_Vars.Data = Data
    Glob_Vars.Target = Target
    Npop = 10
    Chlen = 16
    xmin = matlib.repmat(([0]), Npop, 1)
    xmax = matlib.repmat(([1]), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Crypto
    max_iter = 50

    print('DOA....')
    [bestfit1, fitness1, bestsol1, Time1] = DOA(initsol, fname, xmin, xmax, max_iter)

    print('COA....')
    [bestfit2, fitness2, bestsol2, Time2] = COA(initsol, fname, xmin, xmax, max_iter)

    print('DGO....')
    [bestfit3, fitness3, bestsol3, Time3] = DGO(initsol, fname, xmin, xmax, max_iter)

    print('WPA....')
    [bestfit4, fitness4, bestsol4, Time4] = WPA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

    np.save('Bestsol_Crypto.npy', sol)

# Cryptography
an = 0
if an == 1:
    Eval_all = []
    Data = np.load('Image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    sol = np.load('Bestsol_Crypto.npy', allow_pickle=True)[4, :]
    vl = [1, 2, 3, 4, 5]
    for m in range(len(vl)):
        EVAL = np.zeros((10, 4))
        for i in range(5):
            EVAL[i, :], encrymsg4, decrymsg4 = Model_PLCM(Data, sol[i].astype('int'))
        EVAL[5, :], encrymsg5, decrymsg5 = Model_DES(Data)
        EVAL[6, :], encrymsg6, decrymsg6 = Model_AES(Data)
        EVAL[7, :], encrymsg7, decrymsg7 = Model_ECC(Data)
        EVAL[8, :], encrymsg8, decrymsg8 = Model_PLCM(Data)
        EVAL[9, :], encrymsg9, decrymsg9 = EVAL[4, :], encrymsg4, decrymsg4
        Eval_all.append(EVAL)
    np.save('Eval_time.npy', Eval_all)
    np.save('Encrypted_Data.npy', encrymsg8)
    np.save('Decrypted_Data.npy', decrymsg8)

Plot_Confusion()
Plot_ROC()
plot_results_conv()
Plot_Results_method()
Plot_Results()
Plot_table()
Plot_Results_Feature()
Plot_alg()
plot_KEY()
plot_ENC_DEC_CONS_MEM_1()
plot_CPA_KPA()
