
# Women Cloth Reviews Prediction with Multinomial Naïve Bayes

## Project Overview
This project involves building a machine learning model using the **Multinomial Naïve Bayes** algorithm to predict customer sentiments based on reviews of women's clothing. The focus is on text classification using discrete features, making Multinomial Naïve Bayes particularly suited for this task.

### Dataset
The dataset used in this project is the **Women Clothing E-Commerce Review** dataset, sourced from [YBIFoundation/ProjectHub-MachineLearning](https://raw.githubusercontent.com/YBIFoundation/ProjectHub-MachineLearning/main/Women%20Clothing%20E-Commerce%20Review.csv). This dataset contains customer reviews of women's clothing, which will be used to train and test the sentiment prediction model.

## Requirements
To run the notebook, the following libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required packages using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Steps in the Project
1. **Data Preprocessing**: Loading and cleaning the dataset to ensure it is ready for analysis.
2. **Exploratory Data Analysis (EDA)**: Visualizing key aspects of the data using plots to gain insights.
3. **Feature Engineering**: Processing text data and converting it into features suitable for model training.
4. **Model Training**: Using the Multinomial Naïve Bayes algorithm to train a sentiment prediction model.
5. **Model Evaluation**: Evaluating the performance of the model on a test dataset using metrics such as accuracy, precision, recall, etc.

## Running the Notebook
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd women-cloth-reviews-prediction
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the `INTERNSHIP.ipynb` notebook and run the cells in sequence.

## Results
The model is trained to predict the sentiment of reviews, which can be categorized as positive, negative, or neutral. The performance is evaluated based on accuracy and other classification metrics.

## Future Improvements
- Use more advanced models like **Logistic Regression** or **Random Forest** to improve prediction accuracy.
- Implement **cross-validation** to optimize hyperparameters.
- Experiment with more sophisticated NLP techniques such as **TF-IDF** or **word embeddings**.

## Author
This project was developed as part of an internship application.
