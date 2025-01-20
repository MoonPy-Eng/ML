# Mainframe Memory Usage Predictor

This project is designed to predict CPU usage peaks and anomalies in a mainframe system based on historical memory usage data. It uses machine learning techniques such as clustering and anomaly detection to identify patterns and make predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop an affinity group application for efficient memory allocation using pathfinding principles. The application collects historical data on CPU time, memory usage, and other relevant metrics, preprocesses the data, and uses machine learning models to predict future CPU time peaks. The predictions can then be used to dynamically allocate GPU resources to optimize overall system performance.

## Dataset

The dataset is located at `memory_usage_data.csv` and contains the following columns:
- `Date`: The date of the data point.
- `Time`: The time of the data point.
- `Total_Real_Storage_MB`: Total real storage in MB.
- `Used_Real_Storage_MB`: Used real storage in MB.
- `Free_Real_Storage_MB`: Free real storage in MB.
- `Total_Virtual_Storage_MB`: Total virtual storage in MB.
- `Used_Virtual_Storage_MB`: Used virtual storage in MB.
- `Free_Virtual_Storage_MB`: Free virtual storage in MB.
- `Total_Auxiliary_Storage_GB`: Total auxiliary storage in GB.
- `Used_Auxiliary_Storage_GB`: Used auxiliary storage in GB.
- `Free_Auxiliary_Storage_GB`: Free auxiliary storage in GB.
- `Total_Shared_Memory_Pages`: Total shared memory pages.
- `Used_Shared_Memory_Pages`: Used shared memory pages.
- `Free_Shared_Memory_Pages`: Free shared memory pages.
- `CPU_Time_Sec`: CPU time in seconds.
- `Service_Units`: Service units.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/MoonPy-Eng/ML.git
    cd ML/Unsupervised\ ML/Mainframe-memory-check
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the dataset**: Ensure that the dataset is placed in the same directory as the script with the name `memory_usage_data.csv`.

2. **Run the script**:
    ```bash
    python Peak_Predictor.py
    ```

    The script will read the dataset, preprocess the data, train the machine learning models, and make predictions. The results will be printed to the console.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the APACHE License. See the LICENSE file for more details.