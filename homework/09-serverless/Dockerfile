FROM agrigorev/model-2025-hairstyle:v1

# Install Python libs used in your lambda_handler
RUN pip install pillow onnxruntime numpy

# Copy your lambda code
COPY lambda_function.py .