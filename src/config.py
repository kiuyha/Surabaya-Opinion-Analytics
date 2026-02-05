from supabase import Client
import logging
from typing import Any, Mapping, Dict, Set
from src.utils.types import SearchConfigDict

# Not using log from src/config.py because it will make a circular import
log = logging.getLogger(__name__)

class Config:
    def __init__(self, supabase_client: Client, env: Mapping[str, str]) -> None:
        self.supabase = supabase_client
        self.env = env
        self._hf_token = None
        self._scrape_config = None
        self._unnecessary_hashtags = None
        self._slang_mapping = None
        self._curse_words = None
        self._readme_train_kmeans = None
        self._loc_abbr = None
        self._per_abbr = None
        self._org_abbr = None

    def _get_app_config(self, key: str, default_value: Any = None):
        """
        Fetches a specific configuration value from the 'app_config' table.
        """
        try:
            # Use .eq() to filter by key and .single() to get one record
            response = self.supabase.table('app_config').select('value').eq('key', key).single().execute()
            
            # .single() returns data directly, not in a list
            if response.data:
                # Return the actual value from the 'value' column
                return response.data['value']
            else:
                log.warning(f"App config key '{key}' not found. Using default value.")
                return default_value
                
        except Exception as e:
            log.error(f"Error getting app config for key '{key}': {e}")
            return default_value
    
    @property
    def scrape_config(self) -> SearchConfigDict:
        """
        Fetches the search configuration from Supabase.
        """
        if self._scrape_config is None:
            log.info("Fetching search config from Supabase...")
            response = self._get_app_config(key='scrape-config')
            if response is None:
                raise Exception("Search config not found in Supabase")
            else:
                self._scrape_config = response
        return self._scrape_config
        
    @property
    def unnecessary_hashtags(self) -> Set[str]:
        """
        Fetches the list of unnecessary hashtags to remove in preprocessing from Supabase.
        """
        if self._unnecessary_hashtags is None:
            log.info("Fetching unnecessary hashtags from Supabase...")
            default_value = [
                "#fyp",
                "#viral",
                "#like4like",
                "#viralbanget",
                "#trending",
                "#fypage",
                "#foryoupage",
                "#explore",
                "#instagood",
                "#followforfollow",
                "#f4f",
                "#likeforlike",
                "#l4l",
                "#commentforcomment",
                "#instadaily",
                "#photooftheday",
                "#viralvideo",
                "#xyzbca",
                "#beritaterkini",
                "#terkini",
            ]
            self._unnecessary_hashtags = set(
                self._get_app_config(
                    key='unnecessary-hashtags',
                    default_value=default_value
                )
            )
        return self._unnecessary_hashtags
    
    @property
    def slang_mapping(self) -> Dict[str, str]:
        """
        Fetches the slang mapping dictionary for preprocessing from Supabase.
        """
        if self._slang_mapping is None:
            log.info("Fetching slang mapping from Supabase...")
            default_value = {
                "gimik": "gimmick",
                "emg": "memang",
                "remeh temeh": "remeh",
                "liyu": "pusing",
                "wdym": "what do you mean",
                "mangnya": "memangnya",
                "sisok": "besok",
                "lfg": "let's go",
                "gede": "besar",
                "aja": "saja",
                "kalo": "kalau",
                "yg": "yang",
                "jg": "juga",
                "klo": "kalau",
                "trs" : "terus",
                "lbh": "lebih",
                "gk": "gak",
                "lol": "laugh out loud",
                "lmao": "laugh my ass off",
                "rofl": "rolling on floor laughing",
                "lmfao": "laughing my fucking ass off",
                "roflmao": "rolling on floor laughing my ass off",
            }
            self._slang_mapping = self._get_app_config(
                key='slang-mapping',
                default_value=default_value
            )
            
        return self._slang_mapping
    
    @property
    def curse_words(self) -> Set[str]:
        """
        Fetches the curse words for preprocessing from Supabase.
        """
        if self._curse_words is None:
            log.info("Fetching curse words from Supabase...")
            default_value = [
                "anjing",
                "monyet",
                "anjir",
                "bjir",
            ]
            self._curse_words = set(self._get_app_config(
                key='curse-words',
                default_value=default_value
            ))
            
        return self._curse_words
    
    @property
    def hf_token(self) -> str:
        """
        Gets the Hugging Face token from an environment variable (for Actions)
        or the local folder (for development).
        """
        from huggingface_hub import get_token

        if self._hf_token is None:
            # Prioritize the environment variable for CI/CD environments
            token = self.env.get("HF_TOKEN")
            if token:
                log.info("Found Hugging Face token in environment variable.")
                return token
            
            # Fallback to local token for development
            log.info("No HF_TOKEN env var found, checking local hf folder.")
            try:
                token = HfFolder.get_token()
                if token:
                    log.info("Found Hugging Face token in local folder.")
                    return token
            except Exception:
                pass # Ignore errors if folder doesn't exist
            
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable or run `huggingface-cli login`.")
        
        return self._hf_token

    @property
    def readme_train_kmeans(self) -> str:
        """
        Fetches the readme train kmeans model from Supabase.
        """
        if self._readme_train_kmeans is None:
            log.info("Fetching readme train kmeans model from Supabase...")

            self._readme_train_kmeans = str(self._get_app_config(
                key='readme-train-kmeans',
                default_value=f"""
---
license: mit
language:
- id
tags:
- topic-modeling
- kmeans
- fasttext
- surabaya
---

# Topic Models for Surabaya Tweet Analysis

This repository contains a set of models for performing topic modeling on tweets from Surabaya.

## Models Included

This repository contains two key components:

1.  **`fasttext.model`**: A `gensim` FastText model trained on processed tweet text. It is used to generate semantic vector embeddings for documents.
2.  **`kmeans.joblib`**: A `scikit-learn` K-Means model trained on the vectors produced by the FastText model. It contains the final topic cluster centroids.

## How to Use

Load the models using `gensim`, `joblib`, and `huggingface_hub`.

```python
import joblib
from gensim.models import FastText
from huggingface_hub import hf_hub_download

# Download and load the models
REPO_ID = "{self.env.get("HF_REPO_KMEANS_ID")}"
kmeans_path = hf_hub_download(repo_id=REPO_ID, filename="kmeans.joblib")
hf_hub_download(repo_id=REPO_ID, filename="fasttext.model.wv.vectors_ngrams.npy")
fasttext_path = hf_hub_download(repo_id=REPO_ID, filename="fasttext.model")

kmeans_model = joblib.load(kmeans_path)
fasttext_model = FastText.load(fasttext_path)

print(f"K-Means model loaded with {{kmeans_model.n_clusters}} clusters.")
print(f"FastText model loaded with vector size {{fasttext_model.vector_size}}.")

# You can now use these models for inference.
"""
            ))
        return self._readme_train_kmeans
    
    @property
    def loc_abbr(self) -> Dict[str, str]:
        """
        Fetches the location abbreviations for ner from Supabase.
        """
        if self._loc_abbr is None:
            log.info("Fetching location abbreviations from Supabase...")
            self._loc_abbr = self._get_app_config(
                key='loc-abbr',
                default_value={
                    # Surabaya & East Java Specific
                    "sby": "Surabaya",
                    "sby.": "Surabaya",
                    "kota sby": "Surabaya",
                    "suraboyo": "Surabaya",
                    "suroboyo": "Surabaya",
                    "srby": "Surabaya",
                    "sbaya": "Surabaya",
                    "sda": "Sidoarjo",
                    "sda.": "Sidoarjo",
                    "gsk": "Gresik",
                    "mjk": "Mojokerto",
                    "mlg": "Malang",
                    "malng": "Malang",
                    "btu": "Batu",
                    "lmj": "Lumajang",
                    "jember": "Jember",
                    "bwi": "Banyuwangi",
                    "mdr": "Madura",

                    # Major Indonesian Cities
                    "jkt": "Jakarta",
                    "jkt.": "Jakarta",
                    "dki": "Jakarta",
                    "jaksel": "Jakarta Selatan",
                    "jakbar": "Jakarta Barat",
                    "jaktim": "Jakarta Timur",
                    "jakut": "Jakarta Utara",
                    "jakpus": "Jakarta Pusat",
                    "bdg": "Bandung",
                    "bdg.": "Bandung",
                    "jogja": "Yogyakarta",
                    "yogya": "Yogyakarta",
                    "yk": "Yogyakarta",
                    "diy": "Yogyakarta",
                    "smg": "Semarang",
                    "solo": "Surakarta",
                    "sulo": "Surakarta",
                    "dps": "Denpasar",
                    "bali": "Bali",
                    "mks": "Makassar",
                    "upg": "Makassar",
                    "mdn": "Medan",
                    "plb": "Palembang",
                    "pku": "Pekanbaru",
                    "btm": "Batam",
                    "bpp": "Balikpapan",
                    "smd": "Samarinda",
                    "ikn": "Ibu Kota Nusantara",

                    # Countries
                    "indo": "Indonesia",
                    "ind": "Indonesia",
                    "ina": "Indonesia",
                    "id": "Indonesia",
                    "nkri": "Indonesia",
                    "konoha": "Indonesia", # Slang often found in sentiment analysis
                    "wakanda": "Indonesia", # Slang often found in sentiment analysis
                    "us": "United States",
                    "usa": "United States",
                    "as": "United States",
                    "amerika": "United States",
                    "uk": "United Kingdom",
                    "inggris": "United Kingdom",
                    "sg": "Singapore",
                    "sgp": "Singapore",
                    "singapur": "Singapore",
                    "my": "Malaysia",
                    "malay": "Malaysia",
                    "au": "Australia",
                    "oz": "Australia",
                    "cn": "China",
                    "tiongkok": "China",
                    "rrt": "China",
                    "jp": "Japan",
                    "jepang": "Japan",
                    "kr": "South Korea",
                    "korsel": "South Korea",
                    "korea": "South Korea",
                    "ru": "Russia",
                    "rusia": "Russia",
                    "sa": "Saudi Arabia",
                    "saudi": "Saudi Arabia",
                    "arab": "Saudi Arabia",
                    "uae": "United Arab Emirates",
                    "th": "Thailand",
                    "vn": "Vietnam",
                    "ph": "Philippines",

                    # Major International Cities
                    "nyc": "New York City",
                    "ny": "New York",
                    "la": "Los Angeles",
                    "dc": "Washington D.C.",
                    "lon": "London",
                    "ldn": "London",
                    "kl": "Kuala Lumpur",
                    "tokyo": "Tokyo",
                    "hk": "Hong Kong",
                }
            )
        return self._loc_abbr

    @property
    def per_abbr(self) -> Dict[str, str]:
        """
        Fetches the person abbreviations for ner from Supabase.
        """
        if self._per_abbr is None:
            log.info("Fetching location names from Supabase...")
            self._per_abbr = self._get_app_config(
                key='per-abbr',
                default_value={
                    # Jokowi
                    "jkw": "Jokowi",
                    "jokowi": "Jokowi",
                    "joko widodo": "Jokowi",
                    "jokowidodo": "Jokowi",
                    "mulyono": "Jokowi",
                    "pak jokowi": "Jokowi",
                    "presiden jokowi": "Jokowi",
                    
                    # Prabowo
                    "prabowo": "Prabowo Subianto",
                    "prabowo subianto": "Prabowo Subianto",
                    "pak prabowo": "Prabowo Subianto",
                    "presiden prabowo": "Prabowo Subianto",
                    "wowo": "Prabowo Subianto",
                    "gemoy": "Prabowo Subianto",
                    "08": "Prabowo Subianto",

                    # Gibran
                    "gibran": "Gibran Rakabuming Raka",
                    "gibran rakab": "Gibran Rakabuming Raka",
                    "pak gibran": "Gibran Rakabuming Raka",
                    "wapres gibran": "Gibran Rakabuming Raka",
                    "mas gibran": "Gibran Rakabuming Raka",
                    "gibrann": "Gibran Rakabuming Raka",
                    "samsul": "Gibran Rakabuming Raka", # Slang
                    
                    # Anies
                    "anies": "Anies Baswedan",
                    "anies baswedan": "Anies Baswedan",
                    "anies rasyid baswedan": "Anies Baswedan",
                    "pak anies": "Anies Baswedan",
                    "anis": "Anies Baswedan",
                    "abah": "Anies Baswedan",
                    
                    # Ganjar
                    "ganjar": "Ganjar Pranowo",
                    "ganjar pranowo": "Ganjar Pranowo",
                    "pak ganjar": "Ganjar Pranowo",
                    "gp": "Ganjar Pranowo",
                    
                    # Ridwan Kamil
                    "rk": "Ridwan Kamil",
                    "kang emil": "Ridwan Kamil",
                    "ridwan kamil": "Ridwan Kamil",
                    "emil": "Ridwan Kamil",

                    # Muhaimin Iskandar
                    "cak imin": "Muhaimin Iskandar",
                    "gus imin": "Muhaimin Iskandar",
                    "imin": "Muhaimin Iskandar",
                    "muhaimin": "Muhaimin Iskandar",

                    # Sandiaga Uno
                    "sandi": "Sandiaga Uno",
                    "sandiaga": "Sandiaga Uno",
                    "mas menteri": "Sandiaga Uno",

                    # Mahfud MD
                    "mahfud": "Mahfud MD",
                    "mahfud md": "Mahfud MD",
                    "mmd": "Mahfud MD",

                    # Erick Thohir
                    "erick": "Erick Thohir",
                    "erick thohir": "Erick Thohir",
                    "etho": "Erick Thohir",
                    "et": "Erick Thohir",

                    # Ahok
                    "ahok": "Basuki Tjahaja Purnama",
                    "btp": "Basuki Tjahaja Purnama",
                    "basuki": "Basuki Tjahaja Purnama",

                    # Sri Mulyani
                    "smi": "Sri Mulyani",
                    "sri mulyani": "Sri Mulyani",
                    "bu ani": "Sri Mulyani",
                    
                    # Eri Cahyadi (Surabaya)
                    "eri cahyadi": "Eri Cahyadi",
                    "pak eri": "Eri Cahyadi",
                    "cak eri": "Eri Cahyadi",
                    "wali kota eri": "Eri Cahyadi",
                    "eri" : "Eri Cahyadi",

                    # Armuji (Surabaya Deputy)
                    "armuji": "Armuji",
                    "cak ji": "Armuji",
                    "cak armuji": "Armuji",
                    
                    # Risma
                    "risma": "Tri Rismaharini",
                    "tri rismaharini": "Tri Rismaharini",
                    "bu risma": "Tri Rismaharini",
                    "mensos risma": "Tri Rismaharini",
                    
                    # Khofifah (East Java)
                    "khofifah": "Khofifah Indar Parawansa",
                    "bu khofifah": "Khofifah Indar Parawansa",
                    "bunda indar": "Khofifah Indar Parawansa",
                    
                    # Emil Dardak (East Java)
                    "emil dardak": "Emil Dardak",
                    "mas emil": "Emil Dardak", # Context dependent, could be RK

                    # SBY
                    "sby": "Susilo Bambang Yudhoyono",
                    "susilo bambang yudhoyono": "Susilo Bambang Yudhoyono",
                    "yudhoyono": "Susilo Bambang Yudhoyono",
                    "pak sby": "Susilo Bambang Yudhoyono",
                    "pepo": "Susilo Bambang Yudhoyono",
                    
                    # Megawati
                    "megawati": "Megawati Soekarnoputri",
                    "megawati soekarnoputri": "Megawati Soekarnoputri",
                    "mega": "Megawati Soekarnoputri",
                    "bu mega": "Megawati Soekarnoputri",
                    "ibu mega": "Megawati Soekarnoputri",
                    "ketum pdip": "Megawati Soekarnoputri",
                    
                    # Gus Dur
                    "gus dur": "Abdurrahman Wahid",
                    "gusdur": "Abdurrahman Wahid",
                    "abdurrahman wahid": "Abdurrahman Wahid",
                    "kh abdurrahman wahid": "Abdurrahman Wahid",
                    
                    # Habibie
                    "habibie": "BJ Habibie",
                    "bj habibie": "BJ Habibie",
                    "b.j. habibie": "BJ Habibie",
                    "bacharuddin jusuf habibie": "BJ Habibie",
                    "pak habibie": "BJ Habibie",
                    
                    # Soeharto
                    "soeharto": "Soeharto",
                    "suharto": "Soeharto",
                    "pak harto": "Soeharto",
                    "harto": "Soeharto",
                    "cendana": "Soeharto",
                    
                    # Soekarno
                    "soekarno": "Soekarno",
                    "sukarno": "Soekarno",
                    "bung karno": "Soekarno",
                    "ir. soekarno": "Soekarno",
                    "ir soekarno": "Soekarno",
                    "karno": "Soekarno",
                    "proklamator": "Soekarno",

                    # Hatta
                    "hatta": "Hatta",
                    "bung hatta": "Hatta",
                    "moh. hatta": "Hatta",
                    "moh hatta": "Hatta",
                }
            )
        return self._per_abbr
    
    @property
    def org_abbr(self) -> Dict[str, str]:
        """
        Fetches the person abbreviations for ner from Supabase.
        """
        if self._org_abbr is None:
            log.info("Fetching location names from Supabase...")
            self._org_abbr = self._get_app_config(
                key='org-abbr',
                default_value={
                    # E-commerce & Tech
                    "tokped": "Tokopedia",
                    "tokopedia": "Tokopedia",
                    "toped": "Tokopedia",
                    "shopee": "Shopee",
                    "oren": "Shopee", # Common slang for Shopee
                    "bukalapak": "Bukalapak",
                    "buklapak": "Bukalapak",
                    "bl": "Bukalapak",
                    "blibli": "Blibli",
                    "lazada": "Lazada",
                    "tiktok shop": "TikTok Shop",
                    "tts": "TikTok Shop",
                    "traveloka": "Traveloka",
                    "tiket.com": "Tiket.com",
                    "goto": "GoTo",
                    "google": "Google",
                    "gugel": "Google",
                    "fb": "Facebook",
                    "facebook": "Facebook",
                    "ig": "Instagram",
                    "insta": "Instagram",
                    "instagram": "Instagram",
                    "wa": "WhatsApp",
                    "whatsapp": "WhatsApp",
                    "watsap": "WhatsApp",
                    "x": "X (Twitter)",
                    "twitter": "Twitter",
                    "twt": "Twitter",
                    "yt": "YouTube",
                    "youtube": "YouTube",

                    # Transport & Logistics
                    "gojek": "Gojek",
                    "gocar": "Gojek",
                    "goride": "Gojek",
                    "grab": "Grab",
                    "grabcar": "Grab",
                    "maxim": "Maxim",
                    "indrive": "inDrive",
                    "kai": "KAI",
                    "kereta api": "KAI",
                    "kai access": "KAI Access",
                    "pelni": "Pelni",
                    "transjakarta": "Transjakarta",
                    "tj": "Transjakarta",
                    "transjatim" : "Trans Jatim",
                    "suroboyo bus" : "Suroboyo Bus",
                    "damri": "DAMRI",
                    "jne": "JNE",
                    "jnt": "J&T",
                    "j&t": "J&T",
                    "sicepat": "SiCepat",
                    "anteraja": "Anteraja",
                    "pos": "Pos Indonesia",
                    "pos indonesia": "Pos Indonesia",
                    
                    # Banking & Finance
                    "bi": "Bank Indonesia",
                    "bca": "BCA",
                    "bank central asia": "BCA",
                    "bri": "BRI",
                    "bank rakyat indonesia": "BRI",
                    "bni": "BNI",
                    "mandiri": "Bank Mandiri",
                    "livin": "Bank Mandiri",
                    "bsi": "BSI",
                    "bank syariah indonesia": "BSI",
                    "btn": "BTN",
                    "bjatim": "Bank Jatim",
                    "bank jatim": "Bank Jatim",
                    "ojk": "OJK",
                    "dana": "DANA",
                    "ovo": "OVO",
                    "gopay": "GoPay",
                    "shopeepay": "ShopeePay",
                    "linkaja": "LinkAja",

                    # Government & Institutions
                    "pemkot": "Pemerintah Kota",
                    "pemkot sby": "Pemerintah Kota Surabaya",
                    "pemkab": "Pemerintah Kabupaten",
                    "pdam": "PDAM",
                    "pln": "PLN",
                    "pln123" : "PLN",
                    "pertamina": "Pertamina",
                    "bumn": "BUMN",
                    "dpr": "DPR",
                    "mpr": "MPR",
                    "kpk": "KPK",
                    "mk": "Mahkamah Konstitusi",
                    "ma": "Mahkamah Agung",
                    "ky": "Komisi Yudisial",
                    "kpu": "KPU",
                    "bawaslu": "Bawaslu",
                    "polri": "Polri",
                    "polisi": "Polri",
                    "tni": "TNI",
                    "satpol pp": "Satpol PP",
                    "bpjs": "BPJS",
                    "bpjs kesehatan": "BPJS Kesehatan",
                    "bpjs ketenagakerjaan": "BPJS Ketenagakerjaan",
                    "kementerian": "Kementerian",
                    "kemendagri": "Kementerian Dalam Negeri",
                    "kemenkeu": "Kementerian Keuangan",
                    "kemendikbud": "Kemendikbud",
                    "kemenkes": "Kementerian Kesehatan",
                    "kominfo": "Kominfo",
                    
                    # Telco
                    "telkom": "Telkom Indonesia",
                    "indihome": "IndiHome",
                    "telkomsel": "Telkomsel",
                    "tsel": "Telkomsel",
                    "indosat": "Indosat Ooredoo Hutchison",
                    "isat": "Indosat Ooredoo Hutchison",
                    "im3": "Indosat Ooredoo Hutchison",
                    "xl": "XL Axiata",
                    "axis": "XL Axiata",
                    "smartfren": "Smartfren",
                    "byu": "by.U",

                    # Media
                    "kompas": "Kompas",
                    "detik": "Detikcom",
                    "detikcom": "Detikcom",
                    "tribun": "Tribunnews",
                    "cnn": "CNN Indonesia",
                    "cnn indonesia": "CNN Indonesia",
                    "cnbc": "CNBC Indonesia",
                    "narasi": "Narasi",
                    "tvone": "tvOne",
                    "metro": "Metro TV",
                    
                    # Partai Politik
                    "pdip": "PDI Perjuangan",
                    "pdi-p": "PDI Perjuangan",
                    "banteng": "PDI Perjuangan",
                    "golkar": "Golkar",
                    "beringin": "Golkar",
                    "gerindra": "Gerindra",
                    "demokrat": "Partai Demokrat",
                    "pks": "PKS",
                    "pan": "PAN",
                    "ppp": "PPP",
                    "nasdem": "NasDem",
                    "pkb": "PKB",
                    "psi": "PSI",
                    "mawar": "PSI", # Slang
                    "perindo": "Perindo",
                    "hanura": "Hanura",
                }
            )
        return self._org_abbr