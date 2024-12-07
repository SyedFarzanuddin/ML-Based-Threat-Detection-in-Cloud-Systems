# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Install necessary Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -f https://download.pytorch.org/whl/cpu/torch_stable.html torch==2.2.0+cpu pandas numpy==1.24.4 scikit-learn boto3 

# Add your custom module and Lambda function code to the image
COPY your_module ${LAMBDA_TASK_ROOT}/your_module
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the command to run your Lambda function
CMD ["app.lambda_handler"]
