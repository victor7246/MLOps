from test_onnx import load_onnx_model, get_prediction
from convert_to_onnx import convert_torch_to_onnx

def get_inference(image_path, saved_model_path="pytorch_model_weights.pth"):
    convert_torch_to_onnx(saved_model_path=saved_model_path, onnx_filename="model.onnx")
    ort_session = load_onnx_model("model.onnx")
    return get_prediction(image_path, ort_session)


