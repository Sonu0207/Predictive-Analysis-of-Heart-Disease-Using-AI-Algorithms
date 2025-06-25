## How to Run the Project

### Running the Web Application

To run the web application created for the demonstration, execute the `main.py` program. This will generate a URL that you can open in the browser of your choice. Once opened, enter sample patient information and click the **Predict** button.

> **Note:** The prediction time varies between devices and may take up to a couple of minutes to complete.  
> The prediction label will display either **Not At Risk** or **At Risk** based on the results of the Random Forest model.

### Running the Accuracy Metrics Program

To run the program that prints out our accuracy metrics, execute the `Methods.py` program. This script runs all 5 state-of-the-art (SOTA) methods and outputs the results for each metric along with a confusion matrix table.

### About Commented Code

Large portions of code in `Methods.py` are commented out intentionally. These sections show how we constructed graphs and visuals that helped us improve our models. However, these parts significantly slow down the program and clutter the output, so they are commented out to ensure that only the accuracy metrics are displayed when the program runs.
