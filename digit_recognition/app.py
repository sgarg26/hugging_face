from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 128), nn.Linear(128, num_classes)
        )

    def forward(self, xb):
        return self.classifier(xb)

model2 = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2, torch_dtype=torch.float16)
model2.load_state_dict(torch.load("./data/spam_dectector.pth"))
model2.to("cuda")
model2.eval()

model = Model()
model.load_state_dict(torch.load("data/model.pth"))
model.to("cuda")
model.eval()

app = FastAPI()


async def process_img(img: Image) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.Grayscale(), transforms.PILToTensor()]
    )
    img_tensor = (
        torch.flatten(transforms.functional.invert(transform(img))).to("cuda").float()
    )
    return img_tensor


@app.post("/predict/")
async def predict(image: UploadFile):
    img_data = await image.read()
    img = Image.open(io.BytesIO(img_data))
    img_tensor = await process_img(img)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        print(output)
        prob = nn.functional.softmax(output, dim=0)
        pred = torch.argmax(prob)
        # last_pred.result = {"class": pred.item(), "conf": f"{torch.max(prob)*100:.2f}%"}
    return RedirectResponse(f"http://127.0.0.1:8000/results?pred={int(pred)}")

@app.post("/predict_spam/")
async def predict_spam(text: str):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
        logits = model2(**inputs).logits
    prob = torch.softmax(logits, dim=1)
    preds = torch.argmax(prob, dim=1)
    is_spam = preds.item() == 1
    print('spam' if is_spam else 'not spam')
    return RedirectResponse(f"http://127.0.0.1:8000/results_spam?pred={'spam' if is_spam else 'not spam'}")


@app.post("/results/")
async def results(pred: int):
    content = f"""
    <body>
        Number is a {pred}
    </body>
    """
    return HTMLResponse(content=content)

@app.post("/results_spam/")
async def results_spam(pred: str):
    content = f"""
    <body>
        Text is {pred}
    </body>
    """
    return HTMLResponse(content=content)


@app.get("/get_image/")
async def main():
    content = """
    <body>
        <form action="/predict/" enctype="multipart/form-data" method="post">
            <input name="image" type="file" accept="image/*"><br>
            <input type="submit" value="Go!">
        </form>
    </body>
    """
    return HTMLResponse(content=content)


@app.get("/get_text/")
async def get_text():
    content = """
    <body>
        <form action="/predict_spam/" enctype="multipart/form-data" method="post">
            <input name="text" type="text" value="enter text!"<br>
            <input type="submit" value="Go!">
        </form>
    </body>
    """
    return HTMLResponse(content=content)