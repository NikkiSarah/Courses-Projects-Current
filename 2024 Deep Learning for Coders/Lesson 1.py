from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from fastai.data.external import *
from time import sleep

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(DDGS().images(keywords=term, max_results=max_images)).itemgot('image')

# first search for a single image of a pegasi
urls = search_images('pegasus photos', max_images=1)
print(urls[0])

# download the url and view the image
dest = 'pegasus.jpg'
download_url(urls[0], dest, show_progress=False)

img = Image.open(dest)
img.to_thumb(256, 256)

# repeat with an image of a desert
urls = search_images('desert photos', max_images=1)
dest = 'desert.jpg'
download_url(urls[0], dest, show_progress=False)

img = Image.open(dest)
img.to_thumb(256, 256)

# collect several examples of each
searches = 'pegasus', 'desert'
path = Path('pegasus or not')

for search in searches:
    dest = (path/search)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f"{search} photo"))
    sleep(10)
    download_images(dest, urls=search_images(f'{search} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{search} shade photo'))
    sleep(10)
    resize_images(path/search, max_size=400, dest=path/search)

# check if images have downloaded correctly and remove any that haven't
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(len(failed))

# to train a model, we need DataLoaders - an object containing the
# training and validation sets. It can be created using a DataBlock in
# fastai
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # model inputs are images and the outputs are categories
    get_items=get_image_files, # find all the model inputs i.e. returns a list of image files in a path
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # randomly split the data, retaining 20% for a validation set
    get_y=parent_label, # the label (category) of each file is the name of the parent folder
    item_tfms=[Resize(192, method='squish')] # resize each image by "squishing" rather than cropping
    ).dataloaders(path, bs=32)

# and view a batch of 6 images
dls.show_batch(max_n=6)

# fine-tune a pretrained resnet18 cnn
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)

# what does the model think about the single pegasus image?
is_pegasus, _, probs = learn.predict(PILImage.create('pegasus.jpg'))
print(f"This is a: {is_pegasus}")
print(f"Probability it's a pegasus: {probs[1]:.4f}")
