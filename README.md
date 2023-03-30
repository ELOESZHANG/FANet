# FANet
The main purpose of RGB-T salient object detection (SOD) is to fully integrate and exploit the information from the complementary fusion of modalities to address the underperformance of RGB SOD in some challenging scenes. In this paper, we propose a novel feature aggregation network that can fully mine multi-scale and multi-modal information for complete and accurate RGB-T SOD. Subsequently, a cross-attention fusion module is proposed to adaptively integrate high-level features by using the attention mechanism in the Transformer. Then we design a simple yet effective fast feature aggregation module to fuse low-level features. Through the combined work of the above modules, our network can perform well in some complex scenes by effectively fusing features from RGB and thermal modalities. Finally, several experiments on publicly available datasets such as VT821, VT1000, and VT5000 demonstrate that our method outperforms state-of-the-art methods.
![image](https://github.com/ELOESZHANG/FANet/edit/main/img_demo/Network.jpg)

## Dataset
Download the RGBT dataset here. []

## For training
1. Download the RGBT dataset and put it under the folder datasets.
2. Change the path in train.py.
3. Train model using train.py

## For inference
1. Download the previously trained model here.[]
2. Change the path in RGBT_test.py.
3. Evaluate the model using RGBT_test.py.

## Results
Download the results of our work.[]
