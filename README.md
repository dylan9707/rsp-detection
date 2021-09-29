This is a short homework project practices the NVIDIA "GETTING START WITH JETSON NANO" course https://courses.nvidia.com/courses/course-v1:DLI+S-RX-02+V2/about
and 
"Hello AI world" by Dusty-nv
https://github.com/dusty-nv/jetson-inference.

Thank you guys for your effort in the tutorial, espcially Dusty!

# rsp-detection
Detect Rock-Scissor-Paper using detectnet. You could then play Rock-Scissor-Paper with a computer later! Simple game.

This is done with Jetson-Nano. A tutorial setup can be found:
https://courses.nvidia.com/courses/course-v1:DLI+S-RX-02+V2/about

# Video
An exmple video of this working:
https://www.youtube.com/watch?v=QYX_PFg2zdc

# Data Collection
There are 3 classes in this project: Rock, Scissor, and Paper. I collected 100 pictures for each class because of limited time. Potentially, it should work better with 1000 pictures using different background and lighting. 

The data was collected by manually freeze the frame and draw the bounding box. For each class, 50 pictures are in light while 50 are in dark. Among each of the 50, 10 pictures for each of front, side, and back, 20 are used for tricky angles.

# Training & Export to ONNX
The batch_size and workers are the default setting, 4 and 2, respectively. I ran 30 epoch, which takes around 5 hours (left running overnight).
```
python3 train_ssd.py --dataset-type=voc --data=data/rsp --model-dir=models/rsp --batch-size=4 --workers=2 --epoch=30
```

Export using ONNX:
```
python3 onnx_export.py --model-dir=models/rsp
```

# Running the live feed

After setting up the JetsonNano, docker with the mounted volume
```
docker/run.sh --volume /rsp-detection:/rsp-detection
```
and go to the detection directory
```
cd /jetson-inference/python/training/detection/ssd
```
run
```
detectnet --threshold=0.35 --model=models/rsp/ssd-mobilenet.onnx --labels=models/rsp/labels.txt --input-blob=input_0 --output-cvg
