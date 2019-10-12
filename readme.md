## PhUSE tumor response prediction


* [slides](https://stomioka.github.io/phuse-tumor-ml/docs/tumor_prediction.slides.html)
* [notebook 1](notebooks/01-tumor_prediction.ipynb) 9/27/2019
  - `dummyTGR2.xlsx` used for training/validation/test
* [notebook 2](notebooks/02-tumor_prediction-no-split.ipynb) 10/3/2019
  - `dummyTGR.xlsx`: used for training and validation
  - `tr_rs_final.xls`: used for testing
  - added and ensemble model (majority vote) built with random forest, knn, and xgboost
* [notebook 3](notebooks/03-tumor_prediction-sites-central.ipynb) 10/6/2019
  - Two data sets were used in four ways as shown on the table below and on 5 different algorithms and optimized with random search 3 hold cross validation from 20-30 iterations and 10 different static imputation values for `SUMDIAM`.

    - `tumor0central.xls` -> central
    - `tumor0site.xls` ---> site

      | m | training and validation data |  test data  | test data id |
      |---|------------------------------|-------------|--------------|
      | 1 | central                      | site        | A            |
      | 2 | site                         | central     | B            |
      | 3 | central*85%+site*85%         | central*15% | C            |
      | 3 | central*85%+site*85%         | site*15%    | D            |

* [notebook 4](notebooks/04-goole_colab_other_imputations.ipynb) Google Colab version

  Additional imputation methods are explored to confirm the sensitivity of the imputation method used in [notebook3](notebooks/03-tumor_prediction-sites-central.ipynb).

    - median

    non-parametric approach
    * knn
    * stochastic gradient descent
    * naive bayes
    * decision tree
    * gradient boosting


<details><summary>To be added</summary>
<p>

* [notebook]() (*R*)
* [notebook]() (*Auto ML*)

</p>
</details>
