import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
from openood.networks.clip import clip
from openood.networks.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from openood.networks.clip_for_wordnet_prepare import generate_cossim_idname_wordnet_dedup
import ipdb
import pdb
import json, os

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
all_wnids = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']
imagenet_200_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}
imagenet_r_mask = [wnid in imagenet_200_wnids for wnid in all_wnids] # True or False
imagenet200_classes = []
for i in range(len(imagenet_r_mask)):
    if imagenet_r_mask[i]:
        imagenet200_classes.append(imagenet_classes[i])

def get_class_names(dataset):
    if dataset == 'imagenet':
        return imagenet_classes
    elif dataset == 'imagenet200':
        return imagenet200_classes
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
    # pdb.set_trace()
    if 'simple' in text_prompt:
        return simple_imagenet_template
    elif 'tip' in text_prompt:
        return tip_imagenet_templates
    elif 'vanilla' in text_prompt:
        return ['{}.']
    elif 'nice' in text_prompt:
        return ['The nice {}.']
    elif 'full' in text_prompt:
        return imagenet_templates
    else:
        raise NotImplementedError

# def get_selected_ood_text_list(total_ood_num=1000):
#     wordnet_processed_path = '/home/notebook/code/personal/S9052995/syn_pro/OpenOOD/data/txtfiles/wordnet_imagenet_cossim.pth'
#     wordnet_dict = torch.load(wordnet_processed_path)

#     can_list_adj = wordnet_dict['text_list_adj'] 
#     can_cos_adj = wordnet_dict['cos_sim_adj']
#     can_list_noun = wordnet_dict['text_list_noun'] 
#     can_cos_noun = wordnet_dict['cos_sim_noun'] 

#     # total_ood_num = 1000
#     adj_num = int(total_ood_num * (len(can_list_adj) / (len(can_list_adj) + len(can_list_noun))))
#     noun_num  = total_ood_num - adj_num


#     cate_num = can_cos_adj.size(1)
#     cos_sim_indice_selected = int(cate_num * 0.95)

#     value_cos_adj, _ = can_cos_adj.sort(1) ## small cos sim to large cos sim.
#     value_cos_noun, _ = can_cos_noun.sort(1)

#     selected_value_cos_adj = value_cos_adj[:, cos_sim_indice_selected] ## larger means closer to ID
#     selected_value_cos_noun = value_cos_noun[:, cos_sim_indice_selected] 
#     ## selected the ood can with smaller cos dis to ID.
#     value_sim_adj, indice_sim_adj= selected_value_cos_adj.sort(0) ## smaller to larger
#     value_sim_noun, indice_sim_noun = selected_value_cos_noun.sort(0)
#     # ipdb.set_trace()
#     selected_adj_text = [can_list_adj[i] for i in indice_sim_adj[:adj_num]]
#     selected_noun_text = [can_list_noun[i] for i in indice_sim_noun[:noun_num]]

#     selected_ood_text = selected_adj_text + selected_noun_text
#     return selected_ood_text

def get_selected_ood_text_list(total_ood_num=1000):
    dataset='imagenet'   ## we fix it as imagenet for now; it will be defined in the function input in the revision.
    print('the ID dataset is:', dataset)
    print('total_ood_num is:', total_ood_num)
    #######################################
    foot_path = '/home/notebook/code/personal/S9052995/syn_pro/OpenOOD/data/txtfiles_output/'

    wordnet_processed_path = foot_path + 'wordnet_' + dataset + '_cossim_dedup.pth'
    if os.path.exists(wordnet_processed_path):
        wordnet_dict = torch.load(wordnet_processed_path)
    else:
        classnames = get_class_names(dataset)
        generate_cossim_idname_wordnet_dedup(classnames, wordnet_processed_path)
        wordnet_dict = torch.load(wordnet_processed_path)

    # #######################################  old verision.
    # foot_path = '/home/notebook/code/personal/S9052995/syn_pro/OpenOOD/data/txtfiles/'
    # # selected_adj_noun_path = foot_path + 'selected_adj_noun_0.95_1k.pth'
    # # unselected_adj_noun_path = foot_path + 'unselected_adj_noun_0.95_1k.pth'
    # # wordnet_processed_path = foot_path + 'wordnet_imagenet_cossim.pth'

    # selected_adj_noun_path = foot_path + 'selected_adj_noun_0.95_1k_dedup.pth'
    # unselected_adj_noun_path = foot_path + 'unselected_adj_noun_0.95_1k_dedup.pth'
    # wordnet_processed_path = foot_path + 'wordnet_imagenet_cossim_dedup.pth'
    
    # wordnet_dict = torch.load(wordnet_processed_path)

    can_list_adj = wordnet_dict['text_list_adj'] 
    can_cos_adj = wordnet_dict['cos_sim_adj']
    can_list_noun = wordnet_dict['text_list_noun'] 
    can_cos_noun = wordnet_dict['cos_sim_noun'] 
    

    adj_num = int(total_ood_num * (len(can_list_adj) / (len(can_list_adj) + len(can_list_noun))))
    noun_num  = total_ood_num - adj_num

    cate_num = can_cos_adj.size(1)
    cos_sim_indice_selected = int(cate_num * 0.95)

    value_cos_adj, _ = can_cos_adj.sort(1) ## small cos sim to large cos sim.
    value_cos_noun, _ = can_cos_noun.sort(1)

    selected_value_cos_adj = value_cos_adj[:, cos_sim_indice_selected] ## larger means closer to ID
    selected_value_cos_noun = value_cos_noun[:, cos_sim_indice_selected] 
    ## selected the ood can with smaller cos dis to ID.
    value_sim_adj, indice_sim_adj= selected_value_cos_adj.sort(0) ## smaller to larger, 13101
    value_sim_noun, indice_sim_noun = selected_value_cos_noun.sort(0) ## 58723.

    # pdb.set_trace()
    ## selected the ood can with smaller cos sim to ID, far from ID
    selected_adj_text = [can_list_adj[i] for i in indice_sim_adj[:adj_num]]
    selected_noun_text = [can_list_noun[i] for i in indice_sim_noun[:noun_num]]

    ## also make use of the ood can with large cos sim to ID, near ID; but these data are not used as ood prompt, but use as test points.
    unselected_adj_text = [can_list_adj[i] for i in indice_sim_adj[adj_num:]]
    unselected_noun_text = [can_list_noun[i] for i in indice_sim_noun[noun_num:]]

    ####################### you may want to save the selected words for visualization. 
    # selected_adj_noun_path = foot_path + dataset + '_selected_adj_noun_0.95_'+ str(total_ood_num) + '_dedup.pth'
    # unselected_adj_noun_path = foot_path + dataset + '_unselected_adj_noun_0.95_' + str(total_ood_num) + '_dedup.pth'
    # selected_ood_text = selected_adj_text + selected_noun_text
    # unselected_ood_text = unselected_adj_text + unselected_noun_text

    # save_dict = {
    #     'adj': selected_adj_text,
    #     'noun': selected_noun_text
    # }
    # torch.save(save_dict, selected_adj_noun_path)
    
    # save_dict_unsel = {
    #     'adj': unselected_adj_text,
    #     'noun': unselected_noun_text
    # }
    # torch.save(save_dict_unsel, unselected_adj_noun_path)

    return selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    
    # ipdb.set_trace()
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding ## 77*512
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection ## 512*512
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts):
        # prompts: 1000*77*512
        # tokenized_prompts: 1000*77, it only used to decide the end token. 
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # 1000*77*512

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX # 16 or 4
        OOD_NUM = cfg.OOD_NUM # number of ood prompts
        self.OOD_NUM = OOD_NUM
        ctx_init = cfg.CTX_INIT  # ''
        prompttype = cfg.prompttype
        dtype = clip_model.dtype # torch.float16
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        clip_imsize = clip_model.visual.input_resolution ## 224
        cfg_imsize = cfg.image_size # 224
        # self.ood_virtul_tokenized_prompt = torch.LongTensor(1,77).zero_()
        # self.ood_virtul_tokenized_prompt[:,0] = 1  # most id class return feat at 

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding_temp = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding_temp[0, 1 : 1 + n_ctx, :]
            ood_ctx_vectors = embedding_temp[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
            ood_prompt_prefix = ctx_init
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
            self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
        else:
            # random initialization
            if cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ood_ctx_vectors = torch.empty(OOD_NUM, n_ctx, ctx_dim, dtype=dtype) # OOD_NUM*77*512
                nn.init.normal_(ctx_vectors, std=0.02)
                nn.init.normal_(ood_ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
                ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
            else:
                if prompttype == 'dis_aware':
                    print("Initializing a distribution aware context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 16*512
                    ood_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 16*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ood_ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
                elif prompttype == 'unified':
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 16*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    ood_ctx_vectors = ctx_vectors
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = self.ctx
                elif prompttype == 'class_specific':
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                    ood_ctx_vectors = torch.empty(OOD_NUM, n_ctx, ctx_dim, dtype=dtype) # OOD_NUM*77*512
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ood_ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx+1))
                    self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 16*512
                    self.ood_ctx = nn.Parameter(ood_ctx_vectors)  # to be optimized, 1*77*512
                else:
                    raise NotImplementedError
                # ood_ctx_vectors = torch.empty(1, 77, ctx_dim, dtype=dtype) # 1*77*512
                # ood_ctx_vectors = torch.empty(OOD_NUM, n_ctx+1, ctx_dim, dtype=dtype) # 1*77*512


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # 'X X X X X X X X X X X X X X X X toilet paper.'
        selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(self.OOD_NUM)
        # selected_ood_text = get_selected_ood_text_list(self.OOD_NUM)
        selected_ood_text = selected_adj_text + selected_noun_text
        ood_prompts = [prompt_prefix + " " + name + "." for name in selected_ood_text]
        # ood_prompts = [ood_prompt_prefix + "." for i in range(OOD_NUM)]
        # prompts = prompts + ood_prompts
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 1000*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 1000*77*512
        
        ood_tokenized_prompts = torch.cat([clip.tokenize(p) for p in ood_prompts]) # ood number *77
        with torch.no_grad():
            ood_embedding = clip_model.token_embedding(ood_tokenized_prompts).type(dtype) # 1000*77*512
        # pdb.set_trace()
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS, 1000*1*512
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS, 1000*60*512

        self.register_buffer("ood_token_prefix", ood_embedding[:, :1, :])  # SOS, 
        self.register_buffer("ood_token_suffix", ood_embedding[:, 1 + n_ctx :, :])  # CLS, EOS, 
        self.n_cls = n_cls # 1000
        self.n_ctx = n_ctx # 16
        
        # self.tokenized_prompts = torch.cat((tokenized_prompts, self.ood_virtul_tokenized_prompt), dim=0)  # torch.Tensor, 1001*77
        self.tokenized_prompts = torch.cat((tokenized_prompts, ood_tokenized_prompts), dim=0)  # torch.Tensor, 1001*77
        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor, 1000*77
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION # end

    def forward(self):
        ctx_vanilla = self.ctx # 16*512, parameters to learn.
        ood_ctx_vanilla = self.ood_ctx
        if ctx_vanilla.dim() == 2:
            ctx = ctx_vanilla.unsqueeze(0).expand(self.n_cls, -1, -1) # 100*16*512
            ctx_ood = ood_ctx_vanilla.unsqueeze(0).expand(self.OOD_NUM, -1, -1) # 100*16*512
        else:
            ctx = ctx_vanilla
            ctx_ood = ood_ctx_vanilla

        prefix = self.token_prefix ## 1000*1*512
        suffix = self.token_suffix ## 1000*60*512

        ood_prefix = self.ood_token_prefix ## 1000*1*512
        ood_suffix = self.ood_token_suffix ## 1000*60*512

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            # pdb.set_trace()
            ood_prompts = torch.cat(
                [
                    ood_prefix,  # (n_cls, 1, dim)
                    ctx_ood,     # (n_cls, n_ctx+1, dim)
                    ood_suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts = torch.cat((prompts, ood_prompts), dim=0)
            # prompts = torch.cat((prompts, self.ood_ctx), dim=0) # 1001*77*512

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg.backbone, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vanilla_clip = clip_model.cuda()
        self.text_features = None
        # pdb.set_trace()
        # pdb.set_trace()

    def forward(self, image, return_feat):
        image_features = self.image_encoder(image.type(self.dtype)) ##128*512
        if not self.training and self.text_features is not None:
                text_features = self.text_features  ## accrelating testing. 
            # self.prompts = self.prompt_learner()  # torch.Size([1000, 77, 512])
            # self.tokenized_prompts = self.tokenized_prompts  # 1000*77
            # self.text_features = self.text_encoder(self.prompts, self.tokenized_prompts)
        else:
            # pdb.set_trace()
            print('re-calculate the text feature with learned prompts.')
            prompts = self.prompt_learner() # torch.Size([1000, 77, 512])
            tokenized_prompts = self.tokenized_prompts  ## 1000*77
            text_features = self.text_encoder(prompts, tokenized_prompts) # 1000*512
            self.text_features = text_features
            
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if return_feat:
            return image_features, text_features, logit_scale
        else:
            logits = logit_scale * image_features @ text_features.t()
            return logits


# # https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# def get_text_features(model, dataset, text_prompt):
#     classnames = get_class_names(dataset) # imagenet --> imagenet class names
#     templates = get_templates(text_prompt) # simple --> text prompt for each classes. 
#     print('adopt text prompt of', text_prompt)
#     with torch.no_grad():
#         text_features = []
#         for classname in tqdm(classnames):
#             texts = [template.format(classname) for template in templates]  # format with class
            
#             # if 'cupl' in text_prompt:
#             #     cupl_file = "CuPL_prompts_imagenet.json"
#             #     f = open('./openood/networks/clip/gpt3_prompts/' + cupl_file)
#             #     cupl_prompts = json.load(f)
#             #     texts += cupl_prompts[classname]
#             texts = clip.tokenize(texts).cuda()  # tokenize
#             class_embeddings = model.encode_text(texts)  # embed with text encoder
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # N*C
#             # class_embedding = class_embeddings.mean(dim=0)
#             # class_embedding /= class_embedding.norm()
#             text_features.append(class_embeddings)
#         # pdb.set_trace()
#         text_features = torch.stack(text_features).cuda()
#     return text_features

# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
def get_text_features_neg(model, dataset, text_prompt, text_center, ood_num):
    classnames = get_class_names(dataset) # imagenet --> imagenet class names
    templates = get_templates(text_prompt) # simple --> text prompt for each classes. 
    # templates = ['this is a {} photo.']
    print('adopt text prompt of', text_prompt)
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            if 'cupl' in text_prompt:
                cupl_file = "CuPL_prompts_imagenet.json"
                f = open('./openood/networks/clip/gpt3_prompts/' + cupl_file)
                cupl_prompts = json.load(f)
                texts += cupl_prompts[classname]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.vanilla_clip.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # N*D
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features.append(class_embedding)
            else:
                text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).cuda() # 512*1000

    # mean_text = - text_features.mean(1) # use the mean directly | or the inverse mean.
    # mean_text /= mean_text.norm()
    # mean_text = mean_text.view(-1, 1) 
    selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(total_ood_num=ood_num)
    # adj_imagenet_template = ['this is a {} photo.']
    with torch.no_grad():
        text_features_neg = []
        for classname in tqdm(selected_adj_text):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.vanilla_clip.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features_neg.append(class_embedding)
            else:
                text_features_neg.append(class_embeddings)

        for classname in tqdm(selected_noun_text):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.vanilla_clip.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features_neg.append(class_embedding)
            else:
                text_features_neg.append(class_embeddings)

        text_features_unselected = text_features_neg

        # text_features_unselected = []
        # for classname in tqdm(unselected_adj_text):
        #     texts = [template.format(classname) for template in templates]  # format with class
        #     texts = clip.tokenize(texts).cuda()  # tokenize
        #     class_embeddings = model.vanilla_clip.encode_text(texts)  # embed with text encoder
        #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #     if text_center:
        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         text_features_unselected.append(class_embedding)
        #     else:
        #         text_features_unselected.append(class_embeddings)

        # ################## too many!! activate when used. 
        # for classname in tqdm(unselected_noun_text):
        #     texts = [template.format(classname) for template in templates]  # format with class
        #     texts = clip.tokenize(texts).cuda()  # tokenize
        #     class_embeddings = model.vanilla_clip.encode_text(texts)  # embed with text encoder
        #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #     if text_center:
        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         text_features_unselected.append(class_embedding)
        #     else:
        #         text_features_unselected.append(class_embeddings)

        text_features_unselected = torch.stack(text_features_unselected, dim=1).cuda() # 512*1000  or  torch.Size([7, 10000, 512])
        text_features_neg = torch.stack(text_features_neg, dim=1).cuda() # 512*1000  or  torch.Size([7, 10000, 512])

    text_features = torch.cat((text_features, text_features_neg), dim=1)
    return text_features.transpose(0,1), text_features_unselected.transpose(0,1)

class CoOp_NegOODPrompt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        classnames = get_class_names(cfg.backbone.dataset) # imagenet 
        self.n_cls = len(classnames) 
        # pdb.set_trace()
        self.n_output = self.n_cls + int(cfg.backbone.OOD_NUM)

        # templates = get_templates(cfg.backbone.text_prompt) # simple
        backbone = cfg.backbone.name # 'ViT-B/16'
        assert backbone in clip.available_models()
        # clip_model, self.preprocess = clip.load(backbone, device='cuda')
        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)
        
        self.logit_scale = clip_model.logit_scale.data
        # self.zeroshot_weights = zeroshot_classifier(clip_model, classnames,
        #                                             templates)
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # self.text_features = get_text_features(clip_model.cuda(), cfg.backbone.dataset, cfg.backbone.text_prompt) 
        # print('shape of pre-computed text features:', self.text_features.shape)
        text_center = cfg.backbone.text_center
        self.text_features, self.text_features_unselected = get_text_features_neg(self.model, cfg.backbone.dataset, cfg.backbone.text_prompt, text_center, cfg.backbone.OOD_NUM)
        print('shape of pre-computed text features:', self.text_features.shape)
                

    def forward(self, x, return_feat=False):
        return self.model(x, return_feat)

        # image_features = self.model.encode_image(x)
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ self.zeroshot_weights  # scaling ??
        # return logits

