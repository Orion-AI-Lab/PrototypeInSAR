## Code and Models from the paper "Learning class prototypes from Synthetic InSAR with Vision Transformers "

Work in Progress.

### Trained models

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

### Train from scratch

You can train a new model by executing `main.py` with the proper arguments. Example usage for a model based on the Swin Transformer:

```
python main.py --encoder=swin --synthetic_train_dataset=TRAIN_PATH --synthetic_val_dataset=VALIDATION_PATH --test_dataset=TEST_PATH --batch_size=40
```


### Datasets
The test set C1, as published by [1], can be found [here](https://www.dropbox.com/s/r1duzboualngo08/C1.zip?dl=0).

Additionally, we publish the synthetic data used in this work:
  - [Synthetic train dataset](https://www.dropbox.com/s/hhnfu5bji1lf1ex/PrototypeSyntheticDataset.zip?dl=0)
  - [Synthetic validation Dataset](https://www.dropbox.com/s/mkcvrv3afn9arr1/PrototypeValidationSyntheticDataset.zip?dl=0)



### Citation 
If you use the code or models in this repo cite our paper:

```
@article{bountos2022learning,
  title={Learning class prototypes from Synthetic InSAR with Vision Transformers},
  author={Bountos, Nikolaos Ioannis and Michail, Dimitrios and Papoutsis, Ioannis},
  journal={arXiv preprint arXiv:2201.03016},
  year={2022}
}
```



### References 
[1] Bountos, Nikolaos Ioannis, et al. "Self-supervised contrastive learning for volcanic unrest detection." IEEE Geoscience and Remote Sensing Letters 19 (2021): 1-5.
