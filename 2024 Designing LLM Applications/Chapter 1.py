#%% Training Data Preprocessing
import requests
import justext
from langdetect import detect_langs
from model import KenlmModel

## Data filtering and cleaning
# explore the effect of justext
response = requests.get("https://en.wikipedia.org/wiki/Bodyboarding")
text = justext.justext(response.content, justext.get_stoplist("English"))

boilerplate_text = [content.text for content in text if content.is_boilerplate]
print(boilerplate_text[:50])

useful_text = [content.text for content in text if not content.is_boilerplate]
print(useful_text[:1])

# explore what happens to langdetect if a text switches between 2+ languages
input_text = """Pag-uwi ko galing sa paaralan, sobrang pagod ako dahil sa dami
ng aking ginawa sa buong araw. Ang traffic din sa kalsada, nakaka-stress
talaga! Pero nang makarating ako sa aking tahanan, nabuhayan ako ng loob dahil
sa masarap na amoy ng ulam na inihanda ni nanay. Excited na akong kumain
kasama ang aking pamilya at i-share ang mga kwento ko tungkol sa aking mga
kaibigan, guro, at mga natutunan ko sa school. After dinner, magre-relax muna
ako habang nanonood ng TV, and then magre-review ng lessons bago matulog. Ito
ang routine ko pag-uwi mula sa school, at masaya ako na dumating sa bahay namay
naghihintay na pamilya na handang makinig at suportahan ako sa aking
pag-aaral."""
print(detect_langs(input_text))

input_text = """After a long day at school, pagod na pagod talaga ako. The
traffic on the way home didn't help, nakakastress na nga! But upon arriving
home, I felt a sense of relief dahil sa welcoming atmosphere and the delicious
aroma of the ulam na inihanda ni Mommy. Excited na akong mag-share ng
experiences ko today with my family during dinner, kasama ang mga kwento about
my friends, teachers, and interesting lessons sa school. After eating, it's
time for me to chill while watching some TV shows, and then review my lessons
bago ako matulog. This is my daily routine pag-uwi galing school, and I am
grateful na may loving family ako na handang makinig at supportahan ako sa
aking educational journey."""
print(detect_langs(input_text))

print(detect_langs("I love you too"))


## Selecting quality documents
# load model trained on English wikipedia and get the perplexity
model = KenlmModel.from_pretrained("wikipedia", "en")
model.get_perplexity("""She was a shriveling bumblebee, and he was a bumbling
banshee, but they accepted a position at Gringotts because of their love for
maple syrup""")

