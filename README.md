
# Prevent the Language Model from being Overconfident in Neural Machine Translation

This is the PyTorch implementation of paper: [Prevent the Language Model from being Overconfident in Neural Machine Translation](https://aclanthology.org/2021.acl-long.268.pdf).



We carry out our experiments on standard Transformer with the  [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) toolkit. If you use any source code included in this repo in your work, please cite the following paper.

```
@inproceedings{miao-etal-2021-prevent,
    title = "Prevent the Language Model from being Overconfident in Neural Machine Translation",
    author = "Miao, Mengqi  and
      Meng, Fandong  and
      Liu, Yijin  and
      Zhou, Xiao-Hua  and
      Zhou, Jie",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.268",
    pages = "3456--3468",
}
```




## Runtime Environment
- OS: Ubuntu 16.04.1 LTS 64 bits
- Python version >=3.6
- Pytorch version =1.2

## Training Steps
We use the two-stage training strategy.

### Stage 1: Jointly Pretraining

- Parameters:

  - `-transformer_lm`  our joint training model NMT+LM.
  - `-train_lm` 
  - `-lambda_lm` set to 0.01 for En->De, Zh->En, and En->Fr.
  - `-report_lm` report the accuracy and perplexity scores of the LM during training.

- Run:  

  for example:

  ```
  sh ./train_shells/nmt_lm_ende.sh 
  ```

### Stage 2: Finetuning

- Parameters:
  - `-transformer_lm`  our joint training model NMT+LM.
  - `-fixed_lm`
  - `-add_nmt_loss` use Margin objective or not.
  - `-add_nmt_lm_loss_fn` Margin function M(Δ): Linear, x3 (i.e., Cube),  x5 (i.e., Quintic), Log.
  - `-lambda_add_loss` hyperparameter λ_{M} of Margin loss, set to 5, 8, 8 for EnDe, ZhEn, and EnFr, respectively.
  - `-weight_sentence` use MSO.
  - `-weight_sentence_thresh`  the threshold hyper-parameter k in MSO, set to 0.3, 0.3, 0.4 for EnDe, ZhEn, and EnFr, respectively.

- Run:

  - MTO

    ```
    sh ./train_shells/MTO_ende.sh x5 5.0
    ```

  - MSO

    ```
    sh ./train_shells/MSO_ende.sh x5 5.0 0.3
    ```

    