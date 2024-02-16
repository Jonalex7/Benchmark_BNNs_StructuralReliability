# Benchmarking deep learning methods in a structural reliability context


## How to train a model:
```bash
python main_train.py --method=dropout --lstate=four_branch
```

Available methods: ```bnnbpp, ensembles, dropout, sghmc ```
Available limit state: ```four_branch, himmelblau, electric, parabolic, high_dim```

## :label: TODO 

- [x] MC Dropout - torch version
- [x] SGHMCM - torch version
- [x] Limit states adapted as classes
- [x] Ensemble - torch version
- [x] Bayes by backprop - torch version
- [ ] Active training modalities
- [x] Config and experiments logger 
- [x] Requirements file