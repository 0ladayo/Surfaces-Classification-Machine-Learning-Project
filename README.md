# Surfaces-Classification-Machine-Learning-Project

## Overview
This repository contains a machine learning project focused on the classification of surface types using a Support Vector Machine (SVM) classifier. The dataset was collected from an Inertial Measurement Unit (IMU) mounted to a rollator (a type of mobility aid), with data recorded across five different surface types: Asphalt, Concrete, Grass, Stone, and Tile.

The primary objective of this project is to assist individuals with mobility challenges who use a rollator by providing real-time surface detection. Since the level of difficulty in walking can vary depending on the surface, this system has the potential to enhance safety and comfort by identifying and adapting to different walking conditions.

## Project Structure
- `Exploratory Data Analysis.ipynb`: Jupyter Notebook for initial data exploration, visualization, and understanding.
- `Preprocessing.ipynb`: Jupyter Notebook detailing the data cleaning, transformation, and feature engineering steps.
- `Machine Learning Classification.ipynb`: Jupyter Notebook that implements the SVM classifier, trains the model, and evaluates its performance.
- `datasets/`: Directory containing the data files used in the project.

## Dataset Description
The primary data for this project was collected from an Inertial Measurement Unit (IMU) mounted on a rollator. 3-axes accelerometer and gyroscope data were recorded across five distinct surface types:
- Asphalt
- Concrete
- Grass
- Stone
- Tile

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/0ladayo/Surfaces-Classification-Machine-Learning-Project
   cd Surfaces-Classification-Machine-Learning-Project
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
The project is structured as a sequence of Jupyter Notebooks. It is recommended to run them in the following order:
1. `Exploratory Data Analysis.ipynb`: To understand the dataset.
2. `Preprocessing.ipynb`: To prepare the data for modeling.
3. `Machine Learning Classification.ipynb`: To train and evaluate the classification model.

## Results
Achieved 93% accuracy in classifying 5 surfaces.

## Contributing
Contributions are welcome. Please open an issue or submit a pull request.

## License

> see the [License](LICENSE.txt) file for license rights and limitations (MIT)
