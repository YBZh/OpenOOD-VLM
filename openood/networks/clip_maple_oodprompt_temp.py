import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
from openood.networks.clip_maple_clone import clip
from openood.networks.clip import clip as official_clip
from openood.networks.clip_maple_clone.simple_tokenizer import SimpleTokenizer as _Tokenizer
import ipdb
import pdb
import copy

_tokenizer = _Tokenizer()

imagenet_classes = [
    'tench', 'goldfish', 'great white shark', 'tiger shark',
    'hammerhead shark', 'electric ray', 'stingray', 'rooster', 'hen',
    'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco',
    'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee',
    'American dipper', 'kite (bird of prey)', 'bald eagle', 'vulture',
    'great grey owl', 'fire salamander', 'smooth newt', 'newt',
    'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog',
    'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle',
    'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana',
    'Carolina anole', 'desert grassland whiptail lizard', 'agama',
    'frilled-necked lizard', 'alligator lizard', 'Gila monster',
    'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile',
    'American alligator', 'triceratops', 'worm snake', 'ring-necked snake',
    'eastern hog-nosed snake', 'smooth green snake', 'kingsnake',
    'garter snake', 'water snake', 'vine snake', 'night snake',
    'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
    'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake',
    'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion',
    'yellow garden spider', 'barn spider', 'European garden spider',
    'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
    'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peafowl',
    'quail', 'partridge', 'african grey parrot', 'macaw',
    'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
    'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser',
    'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala',
    'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm',
    'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
    'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab',
    'red king crab', 'American lobster', 'spiny lobster', 'crayfish',
    'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill',
    'flamingo', 'little blue heron', 'great egret', 'bittern bird',
    'crane bird', 'limpkin', 'common gallinule', 'American coot', 'bustard',
    'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher',
    'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale',
    'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese Chin',
    'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon',
    'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound',
    'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound',
    'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound',
    'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet',
    'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki',
    'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
    'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
    'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier',
    'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier',
    'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier',
    'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier',
    'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer',
    'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier',
    'Australian Silky Terrier', 'Soft-coated Wheaten Terrier',
    'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever',
    'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever',
    'Chesapeake Bay Retriever', 'German Shorthaired Pointer', 'Vizsla',
    'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany dog',
    'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel',
    'Cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz',
    'Schipperke', 'Groenendael dog', 'Malinois', 'Briard', 'Australian Kelpie',
    'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie',
    'Border Collie', 'Bouvier des Flandres dog', 'Rottweiler',
    'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher',
    'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
    'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff',
    'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky',
    'Alaskan Malamute', 'Siberian Husky', 'Dalmatian', 'Affenpinscher',
    'Basenji', 'pug', 'Leonberger', 'Newfoundland dog', 'Great Pyrenees dog',
    'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'brussels griffon',
    'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle',
    'Miniature Poodle', 'Standard Poodle',
    'Mexican hairless dog (xoloitzcuintli)', 'grey wolf',
    'Alaskan tundra wolf', 'red wolf or maned wolf', 'coyote', 'dingo',
    'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox',
    'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat',
    'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
    'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear',
    'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle',
    'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle',
    'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant',
    'grasshopper', 'cricket insect', 'stick insect', 'cockroach',
    'praying mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
    'damselfly', 'red admiral butterfly', 'ringlet butterfly',
    'monarch butterfly', 'small white butterfly', 'sulphur butterfly',
    'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
    'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine',
    'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel horse',
    'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox',
    'water buffalo', 'bison', 'ram (adult male sheep)', 'bighorn sheep',
    'Alpine ibex', 'hartebeest', 'impala (antelope)', 'gazelle',
    'arabian camel', 'llama', 'weasel', 'mink', 'European polecat',
    'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo',
    'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
    'siamang', 'guenon', 'patas monkey', 'baboon', 'macaque', 'langur',
    'black-and-white colobus', 'proboscis monkey', 'marmoset',
    'white-headed capuchin', 'howler monkey', 'titi monkey',
    "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur',
    'indri', 'Asian elephant', 'African bush elephant', 'red panda',
    'giant panda', 'snoek fish', 'eel', 'silver salmon', 'rock beauty fish',
    'clownfish', 'sturgeon', 'gar fish', 'lionfish', 'pufferfish', 'abacus',
    'abaya', 'academic gown', 'accordion', 'acoustic guitar',
    'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance',
    'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can',
    'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon',
    'ballpoint pen', 'Band-Aid', 'banjo', 'baluster / handrail', 'barbell',
    'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow',
    'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap',
    'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker',
    'military hat (bearskin or shako)', 'beer bottle', 'beer glass',
    'bell tower', 'baby bib', 'tandem bicycle', 'bikini', 'ring binder',
    'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie',
    'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'hunting bow',
    'bow tie', 'brass memorial plaque', 'bra', 'breakwater', 'breastplate',
    'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train',
    'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe',
    'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit',
    'cardboard box / carton', 'car wheel', 'automated teller machine',
    'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello',
    'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw',
    'storage chest', 'chiffonier', 'bell or wind chime', 'china cabinet',
    'Christmas stocking', 'church', 'movie theater', 'cleaver',
    'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug',
    'coffeemaker', 'spiral or coil', 'combination lock', 'computer keyboard',
    'candy store', 'container ship', 'convertible', 'corkscrew', 'cornet',
    'cowboy boot', 'cowboy hat', 'cradle', 'construction crane',
    'crash helmet', 'crate', 'infant bed', 'Crock Pot', 'croquet ball',
    'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
    'rotary dial telephone', 'diaper', 'digital clock', 'digital watch',
    'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock',
    'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick',
    'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar',
    'electric locomotive', 'entertainment center', 'envelope',
    'espresso machine', 'face powder', 'feather boa', 'filing cabinet',
    'fireboat', 'fire truck', 'fire screen', 'flagpole', 'flute',
    'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen',
    'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat',
    'garbage truck', 'gas mask or respirator', 'gas pump', 'goblet', 'go-kart',
    'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano',
    'greenhouse', 'radiator grille', 'grocery store', 'guillotine',
    'hair clip', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer',
    'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
    'harp', 'combine harvester', 'hatchet', 'holster', 'home theater',
    'honeycomb', 'hook', 'hoop skirt', 'gymnastic horizontal bar',
    'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron',
    'carved pumpkin', 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle', 'rickshaw',
    'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
    'laptop computer', 'lawn mower', 'lens cap', 'letter opener', 'library',
    'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
    'slip-on shoe', 'lotion', 'music speaker', 'loupe magnifying glass',
    'sawmill', 'magnetic compass', 'messenger bag', 'mailbox', 'tights',
    'one-piece bathing suit', 'manhole cover', 'maraca', 'marimba', 'mask',
    'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine cabinet',
    'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
    'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
    'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped',
    'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa',
    'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van',
    'muzzle', 'metal nail', 'neck brace', 'necklace', 'baby pacifier',
    'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer',
    'oil filter', 'pipe organ', 'oscilloscope', 'overskirt', 'bullock cart',
    'oxygen mask', 'product packet / packaging', 'paddle', 'paddle wheel',
    'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel',
    'parachute', 'parallel bars', 'park bench', 'parking meter',
    'railroad car', 'patio', 'payphone', 'pedestal', 'pencil case',
    'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum',
    'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
    'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship',
    'drink pitcher', 'block plane', 'planetarium', 'plastic bag', 'plate rack',
    'farm plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
    'pool table', 'soda bottle', 'plant pot', "potter's wheel", 'power drill',
    'prayer rug', 'printer', 'prison', 'missile', 'projector', 'hockey puck',
    'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket',
    'radiator', 'radio', 'radio telescope', 'rain barrel',
    'recreational vehicle', 'fishing casting reel', 'reflex camera',
    'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle',
    'rocking chair', 'rotisserie', 'eraser', 'rugby ball',
    'ruler measuring stick', 'sneaker', 'safe', 'safety pin', 'salt shaker',
    'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale',
    'school bus', 'schooner', 'scoreboard', 'CRT monitor', 'screw',
    'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store',
    'shoji screen / room divider', 'shopping basket', 'shopping cart',
    'shovel', 'shower cap', 'shower curtain', 'ski', 'balaclava ski mask',
    'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel',
    'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock',
    'solar thermal collector', 'sombrero', 'soup bowl', 'keyboard space bar',
    'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web',
    'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive',
    'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall',
    'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa',
    'submarine', 'suit', 'sundial', 'sunglasses', 'sunglasses', 'sunscreen',
    'suspension bridge', 'mop', 'sweatshirt', 'swim trunks / shorts', 'swing',
    'electrical switch', 'syringe', 'table lamp', 'tank', 'tape player',
    'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof',
    'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof',
    'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
    'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray',
    'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
    'trolleybus', 'trombone', 'hot tub', 'turnstile', 'typewriter keyboard',
    'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase',
    'vaulted or arched ceiling', 'velvet fabric', 'vending machine',
    'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock',
    'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine',
    'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle',
    'hair wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
    'airplane wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence',
    'shipwreck', 'sailboat', 'yurt', 'website', 'comic book', 'crossword',
    'traffic or street sign', 'traffic light', 'dust jacket', 'menu', 'plate',
    'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'popsicle',
    'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog',
    'mashed potatoes', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
    'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber',
    'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith apple',
    'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
    'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara',
    'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
    'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff',
    'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach',
    'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver',
    'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
    'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra',
    'stinkhorn mushroom', 'earth star fungus', 'hen of the woods mushroom',
    'bolete', 'corn cob', 'toilet paper'
]

def get_class_names(dataset):
    if dataset == 'imagenet':
        return imagenet_classes
    else:
        raise NotImplementedError 

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
simple_imagenet_template = ['a photo of a {}.']

############################### tip text prompts
tip_imagenet_templates = [
    'a bad photo of the {}.',
    'a {} in a video game.',
    'a origami {}.',
    'a photo of the small {}.',
    'art of the {}.',
    'a photo of the large {}.',
    'itap of a {}.',
]

def get_templates(text_prompt):
    if 'simple' in text_prompt:
        return simple_imagenet_template
    elif 'tip' in text_prompt:
        return tip_imagenet_templates
    elif 'full' in text_prompt:
        return imagenet_templates
    else:
        raise NotImplementedError

def load_clip_to_cpu_official(backbone_name):
    url = official_clip._MODELS[backbone_name]
    model_path = official_clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = official_clip.build_model(state_dict or model.state_dict())
    
    # ipdb.set_trace()
    return model

def load_clip_to_cpu(backbone_name, cfg):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    # design_details = {"trainer": 'MaPLe'}
    # pdb.set_trace()
    design_details = {"trainer": 'MaPLe',
                    "vision_depth": 0,
                    "language_depth": 0, "vision_ctx": 0,
                    "language_ctx": 0,
                    "maple_length": cfg.backbone.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    
    # ipdb.set_trace()
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # pdb.set_trace()
        n_ctx = cfg.backbone.N_CTX # 16 or 4
        OOD_NUM = cfg.backbone.OOD_NUM # number of ood prompts
        ctx_init = cfg.backbone.CTX_INIT  # ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.backbone.image_size # 224
        assert cfg.backbone.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.backbone.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        self.ood_virtul_tokenized_prompt = torch.LongTensor(OOD_NUM,77).zero_()
        self.ood_virtul_tokenized_prompt[:,0] = 1
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
    
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.backbone.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ood_ctx_vectors = torch.empty(OOD_NUM, 77, ctx_dim, dtype=dtype) # OOD_NUM*77*512
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ood_ctx_vectors = torch.empty(OOD_NUM, 77, ctx_dim, dtype=dtype) # 1*77*512
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ood_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768) # to be optimized
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = torch.cat((tokenized_prompts, self.ood_virtul_tokenized_prompt), dim=0)  # torch.Tensor, 1001*77
        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.backbone.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        prompts = torch.cat((prompts, self.ood_ctx), dim=0) # 1000+OOD_NUM*77*512
        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_feat):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if return_feat:
            return image_features, text_features, logit_scale
        else:
            logits = logit_scale * image_features @ text_features.t()
            return logits


# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
def get_text_features(model, dataset, text_prompt):
    classnames = get_class_names(dataset) # imagenet --> imagenet class names
    templates = get_templates(text_prompt) # simple --> text prompt for each classes. 
    print('adopt text prompt of', text_prompt)
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            ###
            # if 'cupl' in text_prompt:
            #     cupl_file = "CuPL_prompts_imagenet.json"
            #     f = open('./openood/networks/clip/gpt3_prompts/' + cupl_file)
            #     cupl_prompts = json.load(f)
            #     texts += cupl_prompts[classname]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # N*C
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            text_features.append(class_embeddings)
        # pdb.set_trace()
        text_features = torch.stack(text_features).cuda()
    return text_features

class Maple_OneOODPrompt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.OOD_NUM = cfg.backbone.OOD_NUM # number of ood prompts
        classnames = get_class_names(cfg.backbone.dataset) # imagenet 
        # templates = get_templates(cfg.backbone.text_prompt) # simple
        backbone = cfg.backbone.name # 'ViT-B/16'
        assert backbone in clip.available_models()
        # clip_model_official, self.preprocess = clip.load(backbone, device='cuda')
        clip_model_official = load_clip_to_cpu_official(backbone)

        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone, cfg)
        
        self.logit_scale = clip_model.logit_scale.data
        # self.zeroshot_weights = zeroshot_classifier(clip_model, classnames,
        #                                             templates)
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        self.text_features = get_text_features(clip_model_official.cuda(), cfg.backbone.dataset, cfg.backbone.text_prompt) 
        print('shape of pre-computed text features:', self.text_features.shape)
        # pdb.set_trace()

        # print("Turning off gradients in both the image and the text encoder")
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)
                

    def forward(self, x, return_feat=False):
        return self.model(x, return_feat)
