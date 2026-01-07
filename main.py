import streamlit as st
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import time


def get_device():
    # Run everything in CPU, since GPU is not supported by Streamlit.
    return torch.device("cpu")


@st.cache_resource
def load_model():
    vgg16 = models.vgg16_bn().to(get_device())
    vgg16.classifier[-1] = nn.Linear(4096, 4)
    model_path = 'model/VGG16_OCT_Retina_trained_model.pt'
    vgg16.load_state_dict(torch.load(model_path, map_location=get_device()))
    vgg16.eval()
    return vgg16


def load_image(uploaded_file) -> Image:
    return Image.open(uploaded_file).convert("RGB")


def predict(img: Image) -> int:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = Variable(img.unsqueeze(0)).to(get_device())

    vgg16 = load_model()
    with torch.no_grad():
        preds_new_model = vgg16(img)

    _, predicted_class_new_model = torch.max(preds_new_model, 1)
    return int(predicted_class_new_model)



class InformationBox:
    def __init__(self, title: str, description: str, statistics: str="", is_normal: bool=False):
        self.title = title
        self.description = description
        self.statistics = statistics
        self.is_normal = is_normal


information_mapping = [
    InformationBox(
        title="CNV (Choroidal Neovascularization)",
        description="""
CNV is the creation of new, abnormal blood vessels in the choroid layer of the eye (the vascular layer behind the retina).
These vessels break through the barrier (Bruch's membrane) and grow under the retina.
Because these new vessels are fragile and immature, they leak fluid and blood.
This leakage causes the macula to bulge or lift, leading to **rapid** and **severe** central vision loss.
""",
        statistics="""
A meta-analysis estimated that among people aged 45-85, the global prevalence of any AMD is about 8.7% (Vujosevic et al., 2024).
""",
        is_normal=False
    ),
    InformationBox(
        title="DME (Diabetic Macular Edema)",
        description="""
This is a complication of diabetes. High blood sugar levels damage blood vessels in the retina, causing them to leak fluid.
This fluid builds up in the macula, causing it to swell (edema), which leads to blurred or wavy central vision.
""",
        statistics="""
A meta-analysis found the prevalence of DME to be 5.47% in people with diabetes (with some variation among countries) (Jonas et al., 2017).
""",
        is_normal=False
    ),
    InformationBox(
        title="DRUSEN",
        description="""
These are small, yellowish deposits of lipids (fats) and proteins that accumulate under the retina.
While a few small drusen are a normal part of aging, a large number of bigger drusen are a common early sign of age-related macular degeneration (AMD).
""",
        statistics="""
According to a survey of ophthalmology studies, in 2020 there were an estimated 18.8 million people worldwide with DME (Vujosevic et al., 2024).
""",
        is_normal=False
    ),
    InformationBox(
        title="Normal",
        description="Congratulations! Your retina is normal! No diseases detected yet, so keep your healthy lifestyle.",
        is_normal=True
    )
]


def typewriter_effect(text: str, delay: float = 0.01):
    """
    Generates a stream of characters for a typewriter effect.
    """
    for char in text:
        yield char
        time.sleep(delay)


def write_result(result: int):
    information_box = information_mapping[result]
    st.header(information_box.title)
    st.divider()
    st.write_stream(typewriter_effect(information_box.description))
    if information_box.statistics != "":
        st.divider()
        st.write(typewriter_effect(information_box.statistics))
    if information_box.is_normal:
        st.balloons()



def main():
    st.title("OCT Retina Program")
    st.text("This is a simple program to upload OCT Retina image, then detect the disease")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        st.image(img, caption="Uploaded image")

        prediction_result = predict(img)
        write_result(prediction_result)


if __name__ == "__main__":
    main()