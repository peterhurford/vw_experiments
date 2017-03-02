# Examples of Using Vowpal Platypus [![Travis](https://img.shields.io/travis/peterhuford/vp_examples.svg)]()

This is a repository of example code for [Vowpal Platypus](https://github.com/peterhurford/vowpal_platypus), a lightweight Python wrapper for quick, accurate, out-of-core, multi-core machine learning in Python.

Examples:

* **[ALS](als)** -- Run [Alternating Least Squares Collaborative Filtering](https://github.com/JohnLangford/vowpal_wabbit/wiki/Matrix-factorization-example) on different amounts of [MovieLens data](http://grouplens.org/datasets/movielens/). Capable of analyzing 20M product ratings and creating recommendations based on 3,757,811,791 unrated user-product combinations in 21m47s on a 3x c4.8xlarge cluster (108 cores total, 60GB RAM each).

* **[Criteo](criteo)** -- Train on 45.8M rows of user ad clicks (or no clicks) and then test on 6M rows to generate predictions. Gets a log loss of 0.54581 in 14min30sec of training and 8sec of file writing on a m4.10xlarge (40 core 160GB RAM), which is enough to get position #542 on the leaderboard.

* **[NumerAI](numerai)** -- Trains a quick model on [NumerAI data](https://numer.ai/). In December, acheived #432/953 on the leaderboard with log loss 0.68988.

* **[Springleaf](springleaf)** -- Still work in progress.

* **[Titanic](titanic)** -- Compares a variety of models against Titanic survival data from Kaggle. VP achieves a much higher AUC of 0.9574, outperfoming XGBoost in Python (AUC 0.8290) in a similar amount of time, demonstrating that VW is competitive with XGBoost out of the box. Running XGBoost (out of core) and R are work in progress.
