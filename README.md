# Benchmarking deep learning methods in a structural reliability context


## :label: TODO 

- [x] MC Dropout - torch version
- [x] SGHMCM - torch version
- [x] Limit states adapted as classes
- [x] Ensemble - torch version
- [x] Bayes by backprop - torch version
- [ ] Active training modalities
- [ ] Config and experiments logger 
- [ ] Requirements file

to call each method: ```bnnbpp, ensembles, dropout, sghmc ```
to call each limit state: ```four_branch, himmelblau, electric, parabolic, high_dim```

Train a model:
```bash
python main_train.py --method=dropout --lstate=four_branch
```