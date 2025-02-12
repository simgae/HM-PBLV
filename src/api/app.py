from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import threading
import os

# Importiere die YOLO und Faster R-CNN Modelle
from src.models.yolo_v3_model import YoloV3Model
from src.models.faster_rcnn_model import FasterRCNNModel

app = FastAPI()

# Create instances of the models
yolo_model = YoloV3Model()
pretrained_yolo_model = FasterRCNNModel()

# Global variable to keep track of the training status per model
training_in_progress = {"yolo_v3": False, "pretrained_yolo_v3": False}

def check_model_status(model):
    try:
        model.load_model()
        return True
    except:
        return False

# Global variable to track the readiness of models
model_ready_status = {
    "yolo_v3": check_model_status(yolo_model),        # False means model is not trained yet
    "pretrained_yolo_v3": check_model_status(pretrained_yolo_model) # False means model is not trained yet
}



class TrainStatus(BaseModel):
    training: bool
    epoch: Optional[int] = None
    message: Optional[str] = None


@app.post("/start_training/")
async def start_training(epochs: int = Form(1, ge=1), model_type: str = Form("yolo_v3", enum=["yolo_v3","pretrained_yolo_v3"])):
    """
    Start the training of the YOLO v3 model.
    This will run the training in a separate thread to avoid blocking the API.
    """
    global training_in_progress

    if training_in_progress[model_type]:
        raise HTTPException(status_code=400, detail=f"Training for {model_type} is already in progress.")

    def train():
        global training_in_progress
        try:
            training_in_progress[model_type] = True
            if model_type == "yolo_v3":
                yolo_model.train_model(epochs)
            else:
                pretrained_yolo_model.train_model(epochs)
            model_ready_status[model_type] = True
            training_in_progress[model_type] = False
        except Exception as e:
            training_in_progress[model_type] = False
            print(f"Error during training for {model_type}: {e}")

    # Start the training in a new thread so it doesn't block the FastAPI server
    threading.Thread(target=train).start()
    return {"message": f"Training started for {epochs} epochs for {model_type}."}


@app.get("/training_status/")
async def get_training_status(model_type: str = Query("yolo_v3", enum=["yolo_v3", "pretrained_yolo_v3"])):
    """
    Get the current status of the training process.
    """
    if training_in_progress[model_type]:
        return {"training": True, "epoch": None, "message": f"Training for {model_type} is in progress."}
    else:
        return {"training": False, "epoch": None, "message": f"Training for {model_type} is not in progress."}

@app.get("/model_status/")
async def get_model_readiness():
    """
    Get whether each model is ready for use (i.e., has been trained in the past).
    """
    model_readiness = {
        "yolo_v3": {
            "ready": model_ready_status["yolo_v3"],
            "message": "Model is ready for use." if model_ready_status["yolo_v3"] else "Model is not yet trained."
        },
        "pretrained_yolo_v3": {
            "ready": model_ready_status["pretrained_yolo_v3"],
            "message": "Model is ready for use." if model_ready_status["pretrained_yolo_v3"] else "Model is not yet trained."
        }
    }
    return model_readiness

@app.post("/evaluate_image/")
async def evaluate_image(image: UploadFile = File(...), model_type: str = Form("yolo_v3", enum=["yolo_v3", "pretrained_yolo_v3"])):
    """
    Evaluate a single image using the trained YOLO v3 model.
    """
    # Save the uploaded file temporarily
    image_path = f"data/kitti-test/temp_{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())

    if model_type == "yolo_v3":
        model = yolo_model
    else:
        model = pretrained_yolo_model

    # Load the trained YOLO v3 model
    try:
        model.load_model()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {model_type} model: {str(e)}")

    try:
        # Evaluate the image
        model.evaluate_image(image_path)

        # Save the output image in a specific directory
        output_image_path = f"output_images/output_{model_type}.jpg"  # Model writes the image here
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # Assuming `yolo_model.evaluate_image` saves the result to output.jpg

        # Return the URL to the saved output image
        return {
            "message": "Image evaluated successfully.",
            "output_image_url": f"/download_output_image/{model_type}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate image using {model_type}: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)

@app.get("/download_output_image/{model_type}")
async def download_output_image(model_type: str):
    """
    Downloads the output image for the specified model.
    """
    output_image_path = f"output_images/output_{model_type}.jpg"
    if os.path.exists(output_image_path):
        return FileResponse(output_image_path, media_type="image/jpeg", filename=f"output_{model_type}.jpg")
    else:
        raise HTTPException(status_code=404, detail=f"Output image for {model_type} not found.")

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
