import os
import pandas as pd
import numpy as np
import json
import boto3
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from your_module import MultiLabelCNNWithAttention1, MultiLabelCNNWithAttention2

# AWS clients
s3 = boto3.client('s3')
sns = boto3.client('sns')

# S3 Bucket and Model File Paths
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'threatdetection')
MODEL_1_KEY = 'Model_1_state.pth'
MODEL_2_KEY = 'Model_2_state.pth'

# Define attack categories
#attack_categories = ['Analysis', 'Backdoor', 'DoS', 'Exploit', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
attack_categories = []

# Data Preprocessing
def preprocess_model1(df):
    """
    Preprocessing for Model 1.
    Includes feature scaling using StandardScaler.
    """
    label_encoders = {}
    for col in ['proto', 'service', 'state']:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

    if 'attack_cat' in df.columns:
        attack_cat_dummies = pd.get_dummies(df['attack_cat'], prefix='attack_cat')
        df = pd.concat([df, attack_cat_dummies], axis=1)

       # Dynamically define attack categories after df is loaded
        global attack_categories  # Access the global variable
        attack_categories = [col.replace('attack_cat_', '') for col in df.columns if col.startswith('attack_cat_')]
        print(attack_categories)


        df.drop(columns=['attack_cat'], inplace=True)

    # Drop irrelevant columns for Model 1
    drop_cols = [
        'id', 'label', 'attack_cat_Normal', 'attack_cat_Analysis', 
        'attack_cat_Backdoor', 'attack_cat_DoS', 'attack_cat_Exploits', 
        'attack_cat_Fuzzers', 'attack_cat_Generic', 'attack_cat_Reconnaissance', 
        'attack_cat_Shellcode', 'attack_cat_Worms'
    ]
    drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Convert scaled data to PyTorch tensor
    return torch.tensor(scaled_data, dtype=torch.float32)



# Lambda Function
def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name'] if 'Records' in event else None
        key = event['Records'][0]['s3']['object']['key'] if 'Records' in event else None
        local_file = f'/tmp/{key.split("/")[-1]}' if bucket else '/tmp/input.csv'
        if bucket:
            s3.download_file(bucket, key, local_file)
        df = pd.read_csv(local_file)

        processed_data = preprocess_model1(df)
        
        # Ensure that the processed data is numeric
        if not torch.is_tensor(processed_data):
            raise TypeError("Processed data is not a tensor")
        
        # Load models
        local_model1, local_model2 = '/tmp/model1.pth', '/tmp/model2.pth'
        s3.download_file(S3_BUCKET_NAME, MODEL_1_KEY, local_model1)
        s3.download_file(S3_BUCKET_NAME, MODEL_2_KEY, local_model2)
        model1 = MultiLabelCNNWithAttention1.MultiLabelCNNWithAttention1(input_dim=42, output_dim=1)
        model1.load_state_dict(torch.load(local_model1, map_location=torch.device('cpu')))
        model2 = MultiLabelCNNWithAttention2.MultiLabelCNNWithAttention2(input_dim=42, output_dim=9)
        model2.load_state_dict(torch.load(local_model2, map_location=torch.device('cpu')))
        model1.eval(), model2.eval()
        
        # Model 1 prediction for attack detection
        with torch.no_grad():
            predictions = model1(processed_data).numpy() > 0.5
            #print("Model1 Predictions: ", predictions)

        if predictions.sum() == 0:
            return {'statusCode': 200, 'body': json.dumps({'message': 'No threats detected.'})}

        # Select rows where Model1 predicted malicious activity
        malicious_data = df.iloc[np.where(predictions.flatten())[0]]
        print("Malicious data: ", malicious_data)

        # Ensure malicious data contains only numerical columns
        malicious_data = malicious_data.select_dtypes(include=[float, int])
        if malicious_data.empty:
            raise ValueError("Malicious data does not contain any numerical values")

        # Convert to tensor
        malicious_tensor = preprocess_model1(malicious_data)
        #malicious_tensor = torch.tensor(malicious_data.values, dtype=torch.float32)
        #print("malicious_tensor: ", malicious_tensor)

        # Model 2 predictions
        with torch.no_grad():
            attack_probabilities = model2(malicious_tensor).numpy()
            #print("Attack probabilities: ", attack_probabilities)
            print(attack_categories)

                # Extract attack labels
        def get_attack_labels(predictions, threshold=0.5):
            labels = []
            for row in predictions:
                row_labels = []
                for idx, value in enumerate(row):
                    if idx < len(attack_categories) and value > threshold:
                        row_labels.append(attack_categories[idx])
                labels.append(row_labels)
            return labels

        attack_labels = get_attack_labels(attack_probabilities)
        print("Attack labels: ", attack_labels)

        # Format the SNS message and publish before returning
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:445567100649:MaliciousActivityAlerts',
            Message=f"Threats detected: {attack_labels}",
            Subject="Threat Detection Alert"
        )
        print("SNS notification sent.")

        # Prepare final results
        results = {
            "malicious_rows": malicious_data.index.tolist(),
            "attack_labels": attack_labels
        }
        print("Final results: ", results)

        return {'statusCode': 200, 'body': json.dumps({'message': 'Processing completed!', 'malicious_count': len(malicious_data)})}

        
    except Exception as e:
        print(f"Error: {e}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
