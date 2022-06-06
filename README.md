### Code and Models from the paper "Learning from Synthetic InSAR with Vision Transformers: The case of volcanic unrest detection", IEEE Transactions on Geoscience and Remote Sensing, 2022
 
If you use the code or models in this repo cite our paper:

```
@article{bountos2022learning,
  title={Learning class prototypes from Synthetic InSAR with Vision Transformers},
  author={Bountos, Nikolaos Ioannis and Michail, Dimitrios and Papoutsis, Ioannis},
  journal={arXiv preprint arXiv:2201.03016},
  year={2022}
}
```

#### Trained models

You can download the pretrained models [here](https://www.dropbox.com/sh/bnb5ud6gi2bvkcj/AAC5hY4bQG-Nigo_FNzPh3gDa?dl=0).

- The available models are:
  - [Convit-PL](https://www.dropbox.com/s/o4nr7q1ue1l7vpz/convit.pt?dl=0)
  - [DeiT-PL](https://www.dropbox.com/s/h5w7izmrg670r7y/deit.pt?dl=0)
  - [Swin-PL](https://www.dropbox.com/s/btrufmzl8g29yo9/swin.pt?dl=0)
  - [Swin-PL-Pseudo](https://www.dropbox.com/s/3p3da2kyzrbo7xn/SwinPLPseudo.pt?dl=0)


Model usage example:
```
torch.load('swin.pt')
```

#### Train your own model

You can train a new model by executing `main.py` with the proper arguments. The encoder will be automatically initialized with weights pretrained on ImageNet. Example usage for a model based on the Swin Transformer:

```
python main.py --encoder=swin --synthetic_train_dataset=TRAIN_PATH --synthetic_val_dataset=VALIDATION_PATH --test_dataset=TEST_PATH --batch_size=40
```


#### Pseudo training

Based on the models pretrained on the synthetic dataset you can proceed with the pseudo training process by running the `pseudo_training_utils.py` script.
Example: 
```
python pseudo_training_utils.py --unlabeled_path=PATH_OF_UNLABELED_DATASET  --target_path=PATH_TO_STORE_PSEUDOLABELED_SAMPLES --model_root_path=PATH_OF_DOWNLOADED_MODELS --arch=ARCHITECTURE(E.G swin, deit, convit) --test_path=REAL_TEST_PATH --synthetic_validation_path=SYNTHETIC_VAL_PATH
```

#### Datasets

##### Training/validation datasets
The following list contains the synthetic data used in this work:
  - [Synthetic train dataset](https://www.dropbox.com/s/hhnfu5bji1lf1ex/PrototypeSyntheticDataset.zip?dl=0)
  - [Synthetic validation Dataset](https://www.dropbox.com/s/mkcvrv3afn9arr1/PrototypeValidationSyntheticDataset.zip?dl=0)

##### Test dataset
The test set C1, as published by [1], can be found [here](https://www.dropbox.com/s/r1duzboualngo08/C1.zip?dl=0).

##### Unlabeled dataset

The unlabeled dataset used for domain adaptation can be found [here](https://www.dropbox.com/s/gqiw09n21gcbksu/unlabeled.zip?dl=0)





#### References 
[1] Bountos, Nikolaos Ioannis, et al. "Self-supervised contrastive learning for volcanic unrest detection." IEEE Geoscience and Remote Sensing Letters 19 (2021): 1-5.
