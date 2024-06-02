import glob
import os
import shutil
import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from medpy.metric import sensitivity, specificity
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision.transforms import ToTensor, Normalize
from torchvision.models import resnet34


def DataNormalization(originalDir, imageSaveDir, labelSaveDir, ):
    """
        用于从最原始的数据目录中将图像和标签文件移出来并从重新命名
    :param originalDir:
    :param imageSaveDir:
    :param labelSaveDir:
    :return:
    """
    for r, d, f in os.walk(originalDir):
        for i in f:
            if i.endswith("h.nii.gz"):
                print(os.path.join(r, i))
                shutil.copy(os.path.join(r, i), os.path.join(f"{imageSaveDir}/{r.split('/')[-2]}.nii.gz"))
            elif i.endswith("h_image.nii.gz"):
                shutil.copy(os.path.join(r, i), os.path.join(f"{labelSaveDir}/{r.split('/')[-2]}.nii.gz"))


def ConvertNiiToPng(originalDir, saveDir, isLabel=False):
    for filename in os.listdir(originalDir):
        if filename.endswith('.nii.gz'):
            img = sitk.ReadImage(os.path.join(originalDir, filename))
            arr = sitk.GetArrayFromImage(img)
            image = arr[0, :, :] if len(arr.shape) == 3 else arr
            image *= 255 if isLabel else 1
            cv2.imwrite(os.path.join(saveDir, filename.replace('.nii.gz', '.png')), image)
            print(f"Converted {filename} to PNG")


def ConvertImgToNii(imageDir):
    images = sorted(glob.glob(f"{imageDir}/*"))
    for img_path in images:
        print(img_path)
        jpg_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        jpg_image = jpg_image[None, :, :]
        jpg_array = sitk.GetImageFromArray(jpg_image)

        nii_path = os.path.splitext(img_path)[0] + ".nii.gz"
        sitk.WriteImage(jpg_array, nii_path)


def FindBox(img):
    """
        寻找前景包含有效信息的最小矩形区域
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return int(top), int(bottom), int(left), int(right)


def CenterResize(image, mask, expandRate):
    """
        根据给定的扩充倍率和mask位置进行中心裁剪, 而且保证了裁剪的是个正方形, 配合cv2.resize()[改变了分辨率而不是大小] 使用
    """
    top, bottom, left, right = FindBox(mask)
    height = bottom - top + 1
    width = right - left + 1

    expandedHeight = round(height * expandRate)
    expandedWidth = round(width * expandRate)

    if expandedHeight >= expandedWidth:
        sideLength = expandedHeight
        verticalStart = top - round((expandedHeight - height) / 2)
        horzontalStart = left - round((expandedWidth - width) / 2 + (expandedHeight - expandedWidth) / 2)
    else:
        sideLength = expandedWidth
        verticalStart = top - round((expandedHeight - height) / 2 + (expandedWidth - expandedHeight) / 2)
        horzontalStart = left - round((expandedWidth - width) / 2)

    verticalStart = max(verticalStart, 0)
    horzontalStart = max(horzontalStart, 0)

    return np.copy(image[verticalStart:verticalStart + sideLength, horzontalStart:horzontalStart + sideLength]), \
        np.copy(mask[verticalStart:verticalStart + sideLength, horzontalStart:horzontalStart + sideLength])


def CenterResizeDir(images_dir, labels_dir, size=(256, 256), expandRate=1.1):
    """
        对一个目录中所有的文件执行中心裁剪以及重设大小
    """
    images = sorted(glob.glob(f"{images_dir}/*"))
    labels = sorted(glob.glob(f"{labels_dir}/*"))
    for i, j in zip(images, labels):
        print(i)
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(j, cv2.IMREAD_GRAYSCALE)

        new_image, new_label = CenterResize(image, mask, expandRate)

        new_image = cv2.resize(new_image, size)
        new_label = cv2.resize(new_label, size)

        cv2.imwrite(i, new_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(j, new_label, [cv2.IMWRITE_JPEG_QUALITY, 100])


def loadPreTrainModel():
    model = resnet34()
    model.fc = nn.Linear(512, 2, bias=True)
    model.load_state_dict(
        torch.load('/media/sci/P44_Pro/UltraSoundBreastData/实验/深度学习/Data/CheckPoints/预训练检查点/resnet34.bin'))
    # model.fc = nn.Sequential(
    #     nn.Linear(512, 10, bias=True),
    #     nn.ReLU(),
    #     nn.Dropout(),
    #     nn.Linear(10, 2, bias=True),
    # )
    # nn.init.kaiming_normal_(model.fc[0].weight)
    # nn.init.kaiming_normal_(model.fc[3].weight)

    nn.init.kaiming_normal_(model.fc.weight)

    frozen_layers = ["fc", ]
    for name, param in model.named_parameters():
        print(name)
        if not any(layer in name for layer in frozen_layers):
            param.requires_grad = False

    for param in model.parameters():
        print(param.shape, param.requires_grad)

    return model


def predictSingleFIle(model, filePath, ):
    """
        根据模型以及检查点路径, 预测单个文件
    """
    model.eval()
    with torch.no_grad():
        image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = ToTensor()(image)
        image = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)  # 规范化
        image = image[None, :, :, :]  # 添加batch维度

        pred = model(image.cuda())
        pred = torch.argmax(pred, dim=1)

        return pred.item()


def predictDir(checkpointPath, imageDir, labelExcel):
    """
        根据模型检查点加载模型, 预测整个目录
        labelExcel: 到这里去寻找真实的label
    """
    data_dict = pd.read_excel(labelExcel, )
    model = resnet34()
    model.fc = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2, bias=True),
    )
    model.load_state_dict(torch.load(checkpointPath))
    model.to('cuda')

    files = sorted(glob.glob(imageDir + '/*'))
    Labels = []
    predLabels = []
    mappingDict = dict(zip(data_dict["hospital"], data_dict["HER2_low"]))

    for index, filePath in enumerate(files, 1):
        print(f"{index}/{len(files)}, {filePath}")
        patientId = os.path.splitext(os.path.basename(filePath))[0].split('_image')[0]
        label = mappingDict[patientId]
        Labels.append(label)
        predLabels.append(predictSingleFIle(model, filePath))

    print(Labels)
    print(predLabels)

    print(roc_auc_score(Labels, predLabels))  # 计算auc
    print(sensitivity(np.array(Labels), np.array(predLabels)))  # 计算敏感度
    print(specificity(np.array(Labels), np.array(predLabels)))  # 计算特异性
    print(confusion_matrix(np.array(Labels), np.array(predLabels)))  # 根据敏感度和特异性计算混淆矩阵


def extractROISingleFile(imagePath, labelPath, savePath):
    """
        从单张的jpg/png图像中提取出ROI来
    """
    image = cv2.imread(imagePath)
    label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)  # 确保标签是灰度形式

    if image is None or label is None:
        raise FileNotFoundError("指定的图像或标签路径不正确")

    if image.shape[:2] != label.shape:
        raise ValueError("图像和标签尺寸不匹配")

    mask = label == 255

    roi = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    cv2.imwrite(savePath, roi)


def extractROIDir(imageDir, labelDir, saveDir):
    """
        提取整个目录中所有文件的ROI区域到saveDir中
    # """
    imageFiles = sorted(glob.glob(imageDir + '/*'))
    labelFiles = sorted(glob.glob(labelDir + '/*'))
    for image, label in zip(imageFiles, labelFiles):
        print(image)
        fileName = os.path.basename(image)
        extractROISingleFile(image, label, os.path.join(saveDir, fileName))


if __name__ == "__main__":
    pass
