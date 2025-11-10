import onnxruntime as ort
import numpy as np
import torch
import os


def test_onnx_gpu():
    """
    Tests if ONNX Runtime can use the GPU via DirectML or CUDA.
    """
    print(f"ONNX Runtime version: {ort.__version__}")
    available_providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")

    # Determine the best available GPU provider
    # For Windows, onnxruntime-directml provides 'DmlExecutionProvider'
    if 'DmlExecutionProvider' in available_providers:
        provider = 'DmlExecutionProvider'
        print("\nAttempting to use DirectML (DmlExecutionProvider) for GPU acceleration.")
    elif 'CUDAExecutionProvider' in available_providers:
        provider = 'CUDAExecutionProvider'
        print("\nAttempting to use CUDA (CUDAExecutionProvider) for GPU acceleration.")
    else:
        provider = 'CPUExecutionProvider'
        print("\nNo GPU provider found for ONNX Runtime. Falling back to CPU.")

    # Create a simple ONNX model in memory (A + B)
    class SimpleModel(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    model = SimpleModel()
    dummy_input_a = torch.randn(1, 3, 4)
    dummy_input_b = torch.randn(1, 3, 4)
    onnx_model_path = "simple_model.onnx"

    torch.onnx.export(model, (dummy_input_a, dummy_input_b), onnx_model_path,
                      input_names=['A', 'B'], output_names=['C'])

    try:
        # Create an inference session with the selected provider
        sess = ort.InferenceSession(onnx_model_path, providers=[provider])
        print(f"Successfully created ONNX Runtime session with: {sess.get_providers()}")

        # Verify that the intended provider is actually being used
        if provider not in sess.get_providers():
            print(f"\n[ERROR] Failed to initialize session with '{provider}'. The active provider is '{sess.get_providers()[0]}'.")
            return

        # Prepare inputs
        input_a = np.random.randn(1, 3, 4).astype(np.float32)
        input_b = np.random.randn(1, 3, 4).astype(np.float32)
        inputs = {'A': input_a, 'B': input_b}

        # Run inference
        result = sess.run(['C'], inputs)

        # Verify the result
        np.testing.assert_allclose(result[0], input_a + input_b, rtol=1e-5)
        print("\nONNX Runtime GPU test successful! Inference was executed and results are correct.")
        print("Result sample:", result[0][0][0])

    except Exception as e:
        print(f"\nAn error occurred during the ONNX Runtime test: {e}")
    finally:
        # Clean up the dummy model file
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)

if __name__ == "__main__":
    test_onnx_gpu()