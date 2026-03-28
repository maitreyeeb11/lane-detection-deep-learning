import os
import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np


BASE_DATASET_PATH=os.path.join(os.getcwd(),"..","dataset","tusimple_processed")
TEST_IMG_FOLDER=os.path.join(BASE_DATASET_PATH,"test","images")
TEST_MASK_FOLDER=os.path.join(BASE_DATASET_PATH,"test","masks")

print(BASE_DATASET_PATH)
print(TEST_IMG_FOLDER)
print(TEST_MASK_FOLDER)



def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    pred=(pred_mask>threshold).astype(np.uint8)
    gt=(gt_mask>threshold).astype(np.uint8)

    tp = np.logical_and(pred==1, gt==1).sum()
    fp = np.logical_and(pred==1, gt==0).sum()
    fn = np.logical_and(pred==0, gt==1).sum()

    precision=tp/(tp+fp+1e-9)
    recall=tp/(tp+fn+1e-9)
    f1=2*precision*recall/(precision+recall+1e-9)
    iou=tp/(tp+fp+fn+1e-9)

    return precision, recall, f1, iou


#load image
def loadImage(imagePath,maskPath):
    size=[224,224]
    #Read Image 
    image=tf.io.read_file(imagePath)
    #Neural Network operated on tensors
    image=tf.image.decode_jpeg(image,channels=3)
    #resizing of image is needed loading bigger images would require more time and GPU
    image=tf.image.resize(image,size)
    image=image/255.0 #Normalize image 0-1 rather than 0-255

    mask=tf.io.read_file(maskPath)
    mask=tf.image.decode_jpeg(mask,channels=1)
    mask=tf.image.resize(mask,size)
    mask=mask/255.0

    return image,mask

def datasetFromFolder(imageFolder, maskFolder):
    imageFiles=sorted([file for file in os.listdir(imageFolder) if file.endswith('.jpg')])
    maskFiles=sorted([file for file in os.listdir(maskFolder) if file.endswith('.jpg')])
    imagePaths = [os.path.join(imageFolder, f) for f in imageFiles]
    maskPaths  = [os.path.join(maskFolder, f) for f in maskFiles]
    dataset=tf.data.Dataset.from_tensor_slices((imagePaths,maskPaths))
    dataset=dataset.map(lambda imagePath,maskPath: loadImage(imagePath,maskPath))
    dataset=dataset.batch(1)
    return dataset, imageFiles

#Loading the dataset
testDataset, testFiles=datasetFromFolder(TEST_IMG_FOLDER, TEST_MASK_FOLDER)
#Loading the model
model=tf.keras.models.load_model(
    "lane_model_partialfreeze_epoch5.keras",
    compile=False
)

print("Model Loaded Successfully")

#Displaying the image during training 
def displaySample(image, mask, pred):
    plot.figure(figsize=(10,4))
    titles=['Image','Ground Truth Mask','Predicted Mask']
    images=[image,mask,pred]
    for i in range(3):
        plot.subplot(1,3,i+1)
        plot.title(titles[i])
        plot.imshow(tf.keras.preprocessing.image.array_to_img(images[i]))
        plot.axis('off')
    plot.show()

precisions=[]
recalls=[]
f1_scores=[]
ious=[]

#Prediction and Display

print("Running Predictions")
for images, masks in testDataset:
    predMask=model.predict(images, verbose=0)

    # Convert tensors to numpy
    pred_np=predMask[0, :, :, 0]
    gt_np= masks[0, :, :, 0].numpy()

    # Compute metrics
    p, r, f1, iou=compute_metrics(pred_np, gt_np)

    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f1)
    ious.append(iou)

print("\nTEST SET METRICS")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1/Dice: {np.mean(f1_scores):.4f}")
print(f"IoU: {np.mean(ious):.4f}")
