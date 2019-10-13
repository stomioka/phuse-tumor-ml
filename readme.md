## PhUSE tumor response prediction

<details><summary>Notebook 1: ML on dummyTGR2</summary>
<p>

* [slides](https://stomioka.github.io/phuse-tumor-ml/docs/tumor_prediction.slides.html)
* [notebook 1](notebooks/01-tumor_prediction.ipynb) 9/27/2019
  - `dummyTGR2.xlsx` used for training/validation/test
</p>
</details>

<details><summary>Notebook 2: ML on dummy TGR/tr_rs_final</summary>
<p>

* [notebook 2](notebooks/02-tumor_prediction-no-split.ipynb) 10/3/2019
  - `dummyTGR.xlsx`: used for training and validation
  - `tr_rs_final.xls`: used for testing
  - added and ensemble model (majority vote) built with random forest, knn, and xgboost
</p>
</details>

<details><summary>Notebook 3: ML for site & central data</summary>
<p>

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

    - Test Results:

      | m | test data id | metric | rf   | svc  | lr   | knn  | xgb  |
      |---|--------------|--------|------|------|------|------|------|
      | 1 | A            | acc    | 90.1 | 84.5 | 88.1 | 84.6 | 90.4 |
      | 2 | B            | acc    | 83.0 | 73.9 | 78.0 | 81.1 | 82.3 |
      | 3 | C            | acc    | 82.6 | 73.9 | 81.2 | 82.6 | 84.1 |
      | 3 | D            | acc    | **95.2** | 86.5 | 90.3 | 93.3 | **95.2** |
</p>
</details>

<details><summary>Notebook 4: testing imputations</summary>
<p>

* [notebook 4](notebooks/04-other_imputations-google-colab.ipynb) 10/12/2019 Google Colab version

  Additional imputation methods are explored to confirm the sensitivity of the imputation method used in [notebook3](notebooks/03-tumor_prediction-sites-central.ipynb).

  This notebook only implements following imputation for knn model and does not include the testing.

    - median

    **non-parametric approach**

    - knn
    - stochastic gradient descent
    - naive bayes
    - decision tree
    - gradient boosting

    - Test Results:

      | m | test data id | imputation        | rf | svc | lr | knn | xgb |
      |---|--------------|-------------------|----|-----|----|-----|-----|
      | 1 | A            | median            |    |     |    |     |     |
      | 2 | B            | median            |    |     |    |     |     |
      | 3 | C            | median            |    |     |    |     |     |
      | 3 | D            | median            |    |     |    |     |     |
      | 1 | A            | knn               |    |     |    |     |     |
      | 2 | B            | knn               |    |     |    |     |     |
      | 3 | C            | knn               |    |     |    |     |     |
      | 3 | D            | knn               |    |     |    |     |     |
      | 1 | A            | SGD               |    |     |    |     |     |
      | 2 | B            | SGD               |    |     |    |     |     |
      | 3 | C            | SGD               |    |     |    |     |     |
      | 3 | D            | SGD               |    |     |    |     |     |
      | 1 | A            | bayes             |    |     |    |     |     |
      | 2 | B            | bayes             |    |     |    |     |     |
      | 3 | C            | bayes             |    |     |    |     |     |
      | 3 | D            | bayes             |    |     |    |     |     |
      | 1 | A            | decision tree     |    |     |    |     |     |
      | 2 | B            | decision tree     |    |     |    |     |     |
      | 3 | C            | decision tree     |    |     |    |     |     |
      | 3 | D            | decision tree     |    |     |    |     |     |
      | 1 | A            | gradient boosting |    |     |    |     |     |
      | 2 | B            | gradient boosting |    |     |    |     |     |
      | 3 | C            | gradient boosting |    |     |    |     |     |
      | 3 | D            | gradient boosting |    |     |    |     |     |

</p>
</details>
<details><summary>Notebook 5: 4 hidden layer neural network</summary>
<p>

* [notebook 5](notebooks/05-fnn-tumor_prediction-sites-central.ipynb) 10/13/2019

  - [Google colab notebook](notebooks/05_fnn_tumor_prediction_sites_central_google_cola.ipynb)
  - 4 hidden layer neural network for the prediction.
  - Same imputation method as notebook 3

  - Test Results

    | m | training and validation data |  test data  | test data id |acc |loss |
    |---|------------------------------|-------------|--------------|----|-----|
    | 3 | central*85%+site*85%         | central*15% | C            |86.2|0.489|
    | 3 | central*85%+site*85%         | site*15%    | D            |97.1|0.285|
    </p>
    </details>

<details><summary>Notebook 6: cnn+fnn</summary>

<p>

* [notebook 6](notebooks/06_cnn+fnn_tumor_prediction_sites_central_google_cola.ipynb) 10/13/2019
  - Single conv net

    | m | training and validation data |  test data  | test data id |acc |loss |
    |---|------------------------------|-------------|--------------|----|-----|
    | 3 | central*85%+site*85%         | central*15% | C            |87.0|0.404|
    | 3 | central*85%+site*85%         | site*15%    | D            |97.1|0.233|

    </p>
    </details>
    <details><summary>Notebook 7: parallel convolution network</summary>
    <p>
* [notebook 7](notebooks/07_dual_cnn_tumor_prediction_sites_central_google_cola.ipynb) 10/13/2019
  - parallel conv net

    | m | training and validation data |  test data  | test data id |acc |loss |
    |---|------------------------------|-------------|--------------|----|-----|
    | 3 | central*85%+site*85%         | central*15% | C            |86.2|0.380|
    | 3 | central*85%+site*85%         | site*15%    | D            |98.0|0.176|
    </p>
    </details>
