from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
import os
from src.models.yolo_v3_model import YoloV3Model

app = FastAPI()

# Create an instance of the YOLO v3 model
yolo_model = YoloV3Model()

# Global variable to keep track of the training status
training_in_progress = False


class TrainStatus(BaseModel):
    training: bool
    epoch: Optional[int] = None
    message: Optional[str] = None


@app.post("/start_training/")
async def start_training(epochs: int = 1):
    """
    Start the training of the YOLO v3 model.
    This will run the training in a separate thread to avoid blocking the API.
    """
    global training_in_progress

    if training_in_progress:
        raise HTTPException(status_code=400, detail="Training is already in progress.")

    def train():
        global training_in_progress
        try:
            training_in_progress = True
            yolo_model.train_model(epochs)
            training_in_progress = False
        except Exception as e:
            training_in_progress = False
            print(f"Error during training: {e}")

    # Start the training in a new thread so it doesn't block the FastAPI server
    threading.Thread(target=train).start()
    return {"message": f"Training started for {epochs} epochs."}


@app.get("/training_status/")
async def get_training_status():
    """
    Get the current status of the training process.
    """
    if training_in_progress:
        return {"training": True, "epoch": None, "message": "Training is in progress."}
    else:
        return {"training": False, "epoch": None, "message": "Training is not in progress."}


@app.post("/evaluate_image/")
async def evaluate_image(image: UploadFile = File(...)):
    """
    Evaluate a single image using the trained YOLO v3 model.
    """
    # Save the uploaded file temporarily
    image_path = f"data/kitti-test/temp_{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Load the trained YOLO v3 model
    try:
        yolo_model.load_model()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        # Evaluate the image
        yolo_model.evaluate_image(image_path)

        # Return the path to the saved output image
        output_image_path = "output.jpg"  # Model writes the image here
        return {"message": "Image evaluated successfully.", "output_image_path": output_image_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate image: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
