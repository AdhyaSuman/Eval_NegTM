# Evaluating Negative Sampling Approaches for Neural Topic Models

This repository contains the implementation and resources for the paper [Evaluating Negative Sampling Approaches for Neural Topic Models](https://arxiv.org/abs/2503.18167) by [Suman Adhya](https://adhyasuman.github.io/), Avishek Lahiri, Debarshi Kumar Sanyal, and Partha Pratim Das, published in *IEEE Transactions on Artificial Intelligence*, Vol. 5, No. 11, pp. 5630-5642, Nov. 2024.

## ⚡ TL;DR

This work evaluates how different **negative sampling strategies** impact **neural topic models**. By integrating negative samples into the decoder of VAE-based topic models, the authors observe improvements across **topic coherence**, **topic diversity**, and **document classification accuracy**. The study benchmarks multiple strategies across several datasets, showing that **contrastive-style training** enhances the quality and robustness of learned topics—both quantitatively and through human evaluation.

## 📊 Datasets

The following publicly available datasets were used in our experiments. All datasets are preprocessed and available in the repository:

- [20NewsGroups (20NG)](https://github.com/AdhyaSuman/Eval_NegTM/tree/master/preprocessed_datasets/20NewsGroup) — A collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.
- [GoogleNews (GN)](https://github.com/AdhyaSuman/Eval_NegTM/tree/master/preprocessed_datasets/GN) — News headlines and content extracted from Google News, curated for topic modeling tasks.
- [M10](https://github.com/AdhyaSuman/Eval_NegTM/tree/master/preprocessed_datasets/M10) — A scholarly dataset containing abstracts of scientific publications across 10 disciplines.
- [Wiki40B](https://github.com/AdhyaSuman/Eval_NegTM/tree/master/preprocessed_datasets/Wiki40B) — A **subsample** of the [original Wiki40B dataset](https://huggingface.co/datasets/google/wiki40b) consisting subset of 24,774 English documents.


## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{adhya2024evaluating,
  title={Evaluating Negative Sampling Approaches for Neural Topic Models},
  author={Adhya, Suman and Lahiri, Avishek and Sanyal, Debarshi Kumar and Das, Partha Pratim},
  journal={IEEE Transactions on Artificial Intelligence},
  volume={5},
  number={11},
  pages={5630--5642},
  year={2024},
  publisher={IEEE}
}
```

---

## Acknowledgment
All experiments were conducted using **[OCTIS](https://github.com/MIND-Lab/OCTIS)**, an integrated framework for topic modeling, comparison, and optimization.

📌 **Reference:** Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. (2021). *"OCTIS: Comparing and Optimizing Topic Models is Simple!"* [EACL](https://www.aclweb.org/anthology/2021.eacl-demos.31/).

---

## 📬 Contact

For any questions or inquiries, please contact [Suman Adhya](mailto:adhyasuman30@gmail.com).

