from sklearn.datasets import fetch_openml

print("Downloading Elevators dataset...")
# Fetch the dataset directly
elevators = fetch_openml(data_id=216, as_frame=True, parser='auto')

# Get the full dataframe (features + target)
df = elevators.frame

print("Saving to CSV...")
# Save it directly as a CSV file
df.to_csv('elevators_dataset.csv', index=False)

print("Done! Check your folder for 'elevators_dataset.csv'.")