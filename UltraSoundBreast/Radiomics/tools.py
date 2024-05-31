import pandas as pd
import glob
import SimpleITK as sitk
import cv2


def LoadExternalValidation(featuresExcelPath):
    """
        用于从表格中加载外部验证集的数据， 并返回数据和标签
    """
    data = pd.read_excel(featuresExcelPath, )

    X = data.drop(["Target", "Patient"], axis=1)
    Y = data["Target"]

    return X, Y


def CheckShapeMatch(imageDir, labelDir):
    """
        检查图像和标签对应的形状是否相同且都为二维, 通常用在特征提取之前, 就地保存
    """
    images = sorted(glob.glob(imageDir + '/*'))
    labels = sorted(glob.glob(labelDir + '/*'))

    for imagePath, labelPath in zip(images, labels):
        img = sitk.ReadImage(imagePath)
        label = sitk.ReadImage(labelPath)
        imgArr = sitk.GetArrayFromImage(img)
        labelArr = sitk.GetArrayFromImage(label)

        print(imagePath, imgArr.shape, labelArr.shape)

        if len(imgArr.shape) == 3:
            newImg = sitk.GetImageFromArray(imgArr[0])
            sitk.WriteImage(newImg, imagePath)
        elif len(labelArr.shape) == 3:
            newImg = sitk.GetImageFromArray(labelArr[0])
            sitk.WriteImage(newImg, labelPath)
        else:
            continue


if __name__ == "__main__":
    pass
