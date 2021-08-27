# 模型训练
```
cd /home/dff/MFR/FaceRecognition/
python3 script/experiment/mask.py
```

训练完的模型在/ssd_datasets/dff/model/ckpt_best.pth

# 提取图片特征

```
python3 script/experiment/test_single.py
```

提取图片4096维特征，并预测是否佩戴口罩（0表示不戴，1表示戴），结果存为pkl，保存在：/ssd_datasets/gmy/PKU-HumanID/下，"XXX_feat.pkl"为XXX视频的对应结果
