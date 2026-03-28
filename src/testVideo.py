import cv2
import numpy as np
import tensorflow as tf

#Load Model
model = tf.keras.models.load_model("lane_model_partialfreeze_epoch5.keras", compile=False)


VIDEO_PATH=r"D:\LaneDetection\dataset\test2.mp4"  
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: could not open video.")
    exit()

SAVE_OUTPUT = True
if SAVE_OUTPUT:
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out=cv2.VideoWriter("output_lanes.mp4",fourcc,20.0,(1280, 720))


while True:
    ret,frame=cap.read()
    if not ret:
        break

    original=frame.copy()
    h,w,_=frame.shape

    #Resize the image
    image=cv2.resize(frame,(224, 224))
    image=image/255.0
    image=np.expand_dims(image, axis=0)

    # Predict mask
    predMask=model.predict(image)[0]
    predMask=(predMask > 0.5).astype(np.uint8) 
    predMask= cv2.resize(predMask, (w, h))

    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = predMask* 255  # Green channel

    # Overlay lanes
    blended = cv2.addWeighted(original, 1.0, mask_color, 0.7, 0)

    cv2.imshow("Lane Detection", blended)

    if SAVE_OUTPUT:
        out.write(blended)

    #Press q for quiting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
