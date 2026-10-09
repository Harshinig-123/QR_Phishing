# Stage 3: ML Classification (Member 3)

Put all Stage 3 code in this folder (dataset CSV, training script, saved model).

## Your tasks
- Obtain and preprocess the UCI Phishing Websites Dataset
- Understand/select the URL features you can actually compute from a raw URL
- Implement feature extraction for URLs coming from Stage 1 and Stage 2
- Train Random Forest / Logistic Regression, save the model (joblib/pickle)
- Load the model and generate a prediction: Phishing or Legitimate
- Integrate the ML model with the previous stages

## Input
The URL to classify (use Stage 2's `resolved_url` if the URL was shortened).

## Suggested interface
Create `ml_predict.py` with:

```python
def predict_url(url: str) -> dict:
    return {
        "url": url,
        "prediction": "phishing",   # or "legitimate"
        "probability": 0.93,        # model confidence for the predicted class
    }
```

## Heads-up
Many UCI dataset columns (e.g. web traffic, page rank, DNS records) cannot be
computed from a URL string alone. Choose features you can extract, or the
model will not work on real QR codes. Check model/dataset file sizes before
pushing to GitHub.

## Report sections you own
Methodology Stage 3, dataset, feature extraction, ML algorithm,
Tools & Technologies.
