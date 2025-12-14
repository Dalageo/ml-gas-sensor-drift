<div align="center">
  <img src="https://github.com/user-attachments/assets/87911df2-777f-4ac7-8dfb-56f3fda7fc45" alt="Gas sensor" width="600" />
</div>

<div align="center">
  <a href="https://www.python.org/downloads/release/python-3920/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.9.20-blue.svg" alt="Python 3.9.20"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank">
  <img src="https://img.shields.io/badge/CUDA-12.3-brightgreen.svg" alt="CUDA 12.3"></a>
<a href="https://developer.nvidia.com/cudnn" target="_blank">
  <img src="https://img.shields.io/badge/cuDNN-8.9.7-brightgreen.svg" alt="cuDNN 8.9.7"></a>
  <a href="https://github.com/Dalageo/ML-GasSensorDrift/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-800080" alt="License: AGPLv3"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/ML-GasSensorDrift?style=social" alt="GitHub stars">
</div>

# Drift Detection in Gas Sensor Array at Different Concentration Levels ☢️

Sensor calibration is the process of adjusting a sensor's output to ensure it accurately reflects the true value of what it measures. By aligning sensor readings with known standards or reference measurements, calibration corrects deviations caused by manufacturing differences, environmental changes, or sensor drift over time. This process is essential for acquiring reliable data and preventing measurement errors, flawed analyses, and potentially unsafe decisions in systems that depend on precise data.

This project focuses on analyzing the performance of 16 chemical sensors exposed to six different gases at varying concentration levels over time. Using the [Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) dataset, three predictive models were developed: a Decision Tree, a Random Forest, and an XGBoost. Following the recommendation of Irene Rodriguez-Lujan et al. to calibrate sensors across the entire concentration range, including lower concentrations, to ensure optimal accuracy, the three models were developed to predict concentration values within the range of 5 to 1000 ppm based on sensor outputs. They trained on the first six data batches, which are assumed to represent a period when the sensors were properly calibrated. The best-performing model is later applied to data from subsequent batches (7 to 10) to evaluate its performance as sensor characteristics begin to degrade. The main objective was to identify and observe sensor drift, where sensor accuracy decreases over time, and to assess how this drift affects the model's accuracy.

From the data insights, it was determined that distinct models are more effective for each gas group based on their distribution. As a result, three groups were created: Group 1 (Ethanol), Group 2 (Acetaldehyde and Toluene), and Group 3 (Ethylene, Ammonia, and Acetone), which were tuned separately to develop robust models. Among these, the XGBoost model demonstrated the best performance across all groups, achieving low RMSE and MAE metrics on the training set. Its predictions for the later batches indicated sensor drift, as the metrics significantly diverged from those obtained on the test set. **Overall, having a reference set of calibrated data would be highly beneficial for this project, enabling future models to calibrate the sensors effectively.**

*An inconsistency was identified between the documented and actual gas counts within each batch. For example, Batch 1 gas counts are documented as 83 (Ethanol), 30 (Ethylene), 70 (Ammonia), 98 (Acetaldehyde), 90 (Acetone), 74 (Toluene), but the actual order in the dataset is 90 (Ethanol), 98 (Ethylene), 83 (Ammonia), 30 (Acetaldehyde), 70 (Acetone), 74 (Toluene). To address this, an option to align the dataset with the documented gas counts is provided in the notebook. However, it is unclear which order (documented or actual) is correct, so the notebook proceeds with the actual dataset configuration rather than the description.*
 

## Dataset Description

The dataset used in this project is the [Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). It contains 13,910 measurements collected from 16 chemical sensors exposed to six different gases—Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone and Toluene—at concentration levels ranging from 5 to 1000 ppmv. The dataset was gathered over a period of 36 months (January 2008 to February 2011) at a gas delivery platform facility in the ChemoSignals Laboratory at the BioCircuits Institute, University of California, San Diego.

Each of the six district gases is assigned a unique code and tested across specific concentration ranges (in parts per million by volume, ppmv). The table below summarizes the gas names, corresponding codes, and concentration ranges used in the dataset:

| Gas Name      | Gas Code | Concentration Range (ppmv) |
|---------------|----------|----------------------------|
| Ethanol       | 1        | 50 - 1000                 |
| Ethylene      | 2        | 5 - 500                   |
| Ammonia       | 3        | 12 - 1000                 |
| Acetaldehyde  | 4        | 10 - 300                  |
| Acetone       | 5        | 10 - 600                  |
| Toluene       | 6        | 10 - 100                  |

The dataset is divided into ten batches, with each line representing a single measurement. For each measurement, the first value indicates the analyte (coded from 1 to 6 for the different gases), followed by the concentration level in parts per million by volume (ppmv) and a 128-dimensional feature vector representing the sensor responses. This structure enable efficient data processing and model development for a range of tasks, including sensor drift compensation, classification, and regression. Each batch represents a specific period or combination of months, as shown in the table below:

| Batch ID | Month IDs                  | Ethanol | Ethylene | Ammonia | Acetaldehyde | Acetone | Toluene |
|----------|-----------------------------|---------|----------|---------|--------------|---------|---------|
| Batch 1  | Months 1 and 2              | 83      | 30       | 70      | 98           | 90      | 74      |
| Batch 2  | Months 3, 4, 8, 9, and 10   | 100     | 109      | 532     | 334          | 164     | 5       |
| Batch 3  | Months 11, 12, and 13       | 216     | 240      | 275     | 490          | 365     | 0       |
| Batch 4  | Months 14 and 15            | 12      | 30       | 12      | 43           | 64      | 0       |
| Batch 5  | Month 16                    | 20      | 46       | 63      | 40           | 28      | 0       |
| Batch 6  | Months 17 to 20             | 110     | 29       | 606     | 574          | 514     | 467     |
| Batch 7  | Month 21                    | 360     | 744      | 630     | 662          | 649     | 568     |
| Batch 8  | Months 22 and 23            | 40      | 33       | 143     | 30           | 30      | 18      |
| Batch 9  | Months 24 and 30            | 100     | 75       | 78      | 55           | 61      | 101     |
| Batch 10 | Month 36                    | 600     | 600      | 600     | 600          | 600     | 600     |


Every measurement, which is represented by a 128-dimensional feature vector, captures the sensor responses to different gases and includes information about both the maximum resistance change and the response over time. Each of the 16 sensors contributes eight distinct features, creating a standardized feature vector with the following structure:
`DR_1, |DR|_1, EMAi0.001_1, EMAi0.01_1, EMAi0.1_1, EMAd0.001_1, EMAd0.01_1, EMAd0.1_1, ..., DR_16, |DR|_16, EMAi0.001_16, EMAi0.01_16, EMAi0.1_16, EMAd0.001_16, EMAd0.01_16, EMAd0.1_16`

where:

- **Steady-State Features (DR)**: These represent the maximum resistance change relative to the baseline for each sensor, with a normalized version (`|DR|`) also included.
- **Dynamic Features (EMA)**: These features capture the rising and decaying transient portions of the sensor response over time, calculated using an Exponential Moving Average (EMA). For each sensor, EMA features are generated with three smoothing parameters (`Alfa`) set at 0.1, 0.01, and 0.001 to capture the dynamics of both the increasing (EMAi) and decaying (EMAd) parts of the sensor’s response.

## Setup Instructions

### <img src="https://github.com/user-attachments/assets/8d36d1a5-e9b1-40d1-97c9-3d4ca49e9c95" alt="Local PC" width="18" height = "16" /> **Local Environment Setup**

1. **Download the required dataset from**:
    - **[UCI Machine Learning Repository - Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)**
    or, alternatively, if you clone the repository, the dataset will be included directly.

2. **Clone the repository**:
   ```sh
   git clone https://github.com/Dalageo/ml-gas-sensor-drift.git

3. **Navigate to the cloned directory**:
   ```sh
   cd ML-GasSensorDrift
  
4. **Open the `Drift Detection in Gas Sensor Array at Different Concentration Levels.ipynb` using your preferred Jupyter-compatible environment (e.g., [Jupyter Notebook](https://jupyter.org/), [VS Code](https://code.visualstudio.com/), or [PyCharm](https://www.jetbrains.com/pycharm/))**
   
5. **Update the `folder_path` as needed.**
   
6. **Run the cells sequentially to reproduce the results.**

*To run XGBoost on the GPU, you will need to activate GPU support based on your operating system and install the required dependencies. One option is to follow this [guide](https://www.tensorflow.org/install/pip) provided by [TensorFlow](https://www.tensorflow.org/) for detailed instructions.*

## Acknowledgments

The dataset used in this project is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Special thanks to Vergara et al. and Rodriguez-Lujan et al. for making this dataset available for educational and research purposes. Their foundational work is documented in the following publications:

- A. Vergara, S. Vembu, T. Ayhan, M. A. Ryan, M. L. Homer, and R. Huerta, "Chemical gas sensor drift compensation using classifier ensembles," *Sensors and Actuators B: Chemical*, vol. 166–167, pp. 320-329, 2012. [Online]. Available: https://doi.org/10.1016/j.snb.2012.01.074 

- I. Rodriguez-Lujan, J. Fonollosa, A. Vergara, M. Homer, and R. Huerta, "On the calibration of sensor arrays for pattern recognition using the minimal number of experiments," *Chemometrics and Intelligent Laboratory Systems*, vol. 130, pp. 123-134, 2014. [Online]. Available: https://doi.org/10.1016/j.chemolab.2013.10.012

## License

The dataset included in this repository is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/), while the notebook and accompanying documentation are licensed under the [AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html). AGPL-3.0 license was chosen to promote open collaboration, ensure transparency, and allow others to freely use, modify, and contribute to the work, while ensuring that any improvements or modifications remain accessible to everyone under the same license.

<div align="center">
  <br>
  <a href="https://creativecommons.org/licenses/by/4.0">
    <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by.svg" width="170"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://www.gnu.org/licenses/agpl-3.0.en.html">
    <img src="https://github.com/user-attachments/assets/f3c6face-aa86-45da-8d20-d8ae25e49e28" alt="AGPLv3-Logo" width="200">
  </a>
</div>














