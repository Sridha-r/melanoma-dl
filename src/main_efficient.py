import torch
import pandas as pd


def generate_predictions(
    model,
    test_loader,
    device
):
    model.eval()

    predictions = []

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)

            predictions.extend(
                probs.cpu().numpy().ravel()
            )

    return predictions


def create_submission(
    test_df,
    predictions,
    output_path="submission.csv"
):
    submission = pd.DataFrame({
        "image_name": test_df["image_name"].values,
        "target": predictions
    })

    submission.to_csv(
        output_path,
        index=False
    )

    print("Submission created!")
    print(submission.head())
    print("Shape:", submission.shape)

    return submission