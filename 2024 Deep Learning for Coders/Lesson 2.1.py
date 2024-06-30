from fastai.vision.all import *
import gradio as gr

# view an image
path = Path('HFspaceAnimalImageClassifier')
file = path/'15ee4b08-eb8b-4ce9-808c-c821189ce8c0.jpg'
img = PILImage.create(file)
img.to_thumb(192, 192)

# load the saved model and predict the class of an image
learn = load_learner('model.pkl')
learn.predict(img)

# gradio requires a dictionary of the possible classes and the probability the
# image belongs to that class
categories = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

classify_image(img)

# create a gradio interface
image = gr.Image()
label = gr.Label()
examples = [path/'lion.jpg', path/'boerperd.jpg', path/'hippo.jpg',
            path/'cheetah.jpg', path/'rhino.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
