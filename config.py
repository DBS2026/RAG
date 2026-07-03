"""
Central configuration for the Datasheet Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def require_gemini_key():
    """
    Fails loudly and immediately with a clear message if GEMINI_API_KEY is
    missing, instead of letting a bare Gemini client call fail deep inside
    an API request. app.py checks this before importing backend modules;
    this is the defense-in-depth version for any other entry point
    (scripts, tests, a future FastAPI backend) that imports generator.py
    or vectorspace.py directly.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable, or in a "
            ".env file (see .env.example), before using any module that calls the "
            "Gemini API. Get a free key at https://aistudio.google.com/apikey"
        )

# --- Model names ---
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# --- Storage paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(BASE_DIR, "data", "pages")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_cache.sqlite")
ANSWER_CACHE_MAX_ITEMS = 200

# --- Chunking (Optimized for Dense Technical Manuals) ---
MAX_CHUNK_CHARS = 700          
CHUNK_OVERLAP_CHARS = 100       
MIN_SECTION_CHARS = 120         

SECTION_HEADING_PATTERNS = [
    r"Absolute Maximum Ratings",
    r"Recommended Operating Conditions",
    r"Electrical Characteristics",
    r"Thermal (?:Information|Characteristics|Resistance)",
    r"Pin (?:Description|Configuration|Assignment|Out|Definitions|Layout)s?",
    r"GPIO Matrix",
    r"IO MUX",
    r"Signal Description",
    r"Terminal Functions",
    r"Connection Diagram",
    r"(?:Functional )?Block Diagram",
    r"Typical Application(?: Circuit)?s?",
    r"Application (?:Information|Circuit|Note)s?",
    r"Timing (?:Diagram|Characteristics)",
    r"Package (?:Information|Drawing|Outline)s?",
    r"Mechanical (?:Data|Drawing|Information)",
    r"Ordering Information",
    r"Typical (?:Performance )?Characteristics",
    r"Detailed Description",
    r"Feature[s]?(?:\s+and\s+Benefits)?",
    r"Overview",
    r"Revision History",
    r"General Description",
    r"Reference Design[s]?",
    r"(?:Module )?Schematics?",
    r"PCB Layout(?: Guidelines)?",
    r"Hardware Design(?: Guidelines)?",
    r"Boot (?:Configuration|Mode|Pins?|Strapping)",
    r"Strapping Pins?",
    r"(?:Power|Current) Consumption",
    r"Application Example[s]?",
]

# --- Retrieval Scales ---
# Engineering questions like "design a minimum circuit" need context spread
# across 5-8 pages (schematic, electrical characteristics, PCB layout,
# operating conditions, boot pins). TOP_K=5 was cutting that off early.
TOP_K_RESULTS = 8
CANDIDATE_POOL_SIZE = 25
RRF_K = 60
VECTOR_WEIGHT = 0.45
BM25_WEIGHT = 0.55

# Cap how many chunks from the same page can occupy the final result set,
# so one heavily-matched page doesn't crowd out other relevant pages —
# e.g. 5 chunks from page 22 and none from the schematic on page 32.
MAX_CHUNKS_PER_PAGE = 2

# --- Query Decomposition ---
ENABLE_QUERY_DECOMPOSITION = True
# Skip the extra LLM call for short/simple questions — only decompose
# questions that look like they're asking about several distinct things.
DECOMPOSITION_MIN_CHARS = 45
MAX_SUBQUERIES = 6
TOP_K_PER_SUBQUERY = 4          # per-subquery retrieval depth before merge
DECOMPOSITION_MODEL = os.getenv("DECOMPOSITION_MODEL", GENERATION_MODEL)

# --- Multimodal Figure Summaries ---
ENABLE_FIGURE_SUMMARIES = True
FIGURE_SUMMARY_MODEL = os.getenv("FIGURE_SUMMARY_MODEL", GENERATION_MODEL)
FIGURE_SUMMARY_MAX_CHARS = 500
FIGURE_SUMMARY_CACHE_PATH = os.path.join(CACHE_DIR, "figure_summary_cache.sqlite")

# --- Structured Table Extraction ---
ENABLE_STRUCTURED_TABLE_FALLBACK = True
TABLE_STRUCTURING_MODEL = os.getenv("TABLE_STRUCTURING_MODEL", GENERATION_MODEL)
# A pdfplumber table is considered "low quality" (merged-cell / multi-page
# artifacts) if more than this fraction of its cells are empty, or if it
# looks like a truncated continuation, triggering the vision fallback.
TABLE_EMPTY_CELL_RATIO_THRESHOLD = 0.35
TABLE_CONTINUATION_MARKERS = [
    r"\(continued\)", r"continued from previous page", r"table\s+\d+[A-Za-z]?\s*\(cont",
]

# --- Advanced Hardware Synonym Dictionary ---
# Single-word triggers: matched with a word-boundary regex against the question.
QUERY_SYNONYMS = {
    "pinout": ["pin description", "pin configuration", "pin assignment", "pinout", "gpio", "gnd", "pin layout", "package pinout", "terminal functions"],
    "pin": ["pin description", "pin configuration", "pin assignment", "pinout", "gpio", "gnd", "pin definitions", "terminal functions"],
    "gpio": ["pin description", "pin configuration", "pin assignment", "pinout", "io", "gnd", "gpio matrix", "io mux"],
    "voltage": ["vin", "vout", "vdd", "3.3v", "5v", "absolute maximum ratings", "electrical characteristics"],
    "vin": ["voltage", "input voltage", "vdd", "absolute maximum ratings"],
    "current": ["icc", "supply current", "quiescent", "current consumption", "electrical characteristics"],
    "spi": ["serial peripheral interface", "sclk", "miso", "mosi", "ss_b"],
    "i2c": ["twi", "two-wire", "sda", "scl"],
    "uart": ["serial", "tx", "rx", "baud rate"],
    "adc": ["analog", "analog input", "converter", "resolution"],
    "flash": ["memory", "rom", "nvm", "eeprom"],
    "reset": ["chip_en", "enable", "rst", "power-down"],
    "clock": ["xtal", "oscillator", "clk", "frequency", "clock tree"],
    "temperature": ["thermal information", "thermal resistance", "absolute maximum ratings", "operating temperature"],
    # --- Added hardware vocabulary ---
    "boot": ["strap", "strapping", "startup", "power-up", "boot configuration", "boot mode", "chip_en", "reset"],
    "strap": ["strapping", "boot", "boot configuration", "gpio"],
    "startup": ["power-up", "boot", "boot mode", "reset"],
    "schematic": ["reference design", "reference schematic", "application circuit", "typical application", "module schematics"],
    "pcb": ["layout", "pcb layout", "land pattern", "footprint", "hardware design"],
    "layout": ["pcb layout", "hardware design", "land pattern", "footprint"],
    "rf": ["antenna", "wifi", "bluetooth", "rf layout", "matching network"],
    "antenna": ["rf", "matching network", "rf layout"],
    "wifi": ["rf", "antenna", "wireless", "802.11"],
    "bluetooth": ["rf", "antenna", "wireless", "ble"],
    "psram": ["flash", "memory", "external memory"],
    "xtal": ["crystal", "clock", "oscillator", "clock tree"],
    "crystal": ["xtal", "clock", "oscillator", "clock tree"],
    "sleep": ["deep sleep", "low power", "power modes", "rtc"],
    "rtc": ["real-time clock", "sleep", "deep sleep", "low power"],
    "package": ["dimensions", "land pattern", "package outline", "mechanical data"],
    "dimensions": ["package", "mechanical data", "land pattern"],
    "esd": ["electrostatic discharge", "absolute maximum ratings", "protection"],
    "decoupling": ["bypass capacitor", "decoupling capacitor", "power supply", "reference design"],
}

# Phrase-level triggers: matched by substring search rather than a single
# word, since queries like "minimum circuit" or "reference design" carry
# meaning as a phrase that a per-word synonym lookup would miss entirely.
PHRASE_SYNONYMS = {
    "minimum circuit": ["reference schematic", "reference design", "application circuit",
                         "typical application", "power supply", "decoupling capacitor",
                         "chip_en", "boot", "pcb layout", "hardware design"],
    "reference design": ["reference schematic", "typical application", "application circuit",
                          "hardware design", "pcb layout"],
    "typical application": ["reference design", "application circuit", "reference schematic"],
    "application circuit": ["reference design", "typical application", "reference schematic"],
    "power consumption": ["current consumption", "sleep current", "low power", "power modes"],
    "current consumption": ["power consumption", "icc", "supply current", "quiescent"],
    "boot mode": ["boot configuration", "strapping pins", "boot pins", "chip_en"],
    "boot pins": ["strapping pins", "boot configuration", "boot mode"],
    "low power": ["sleep", "deep sleep", "power modes", "rtc", "power consumption"],
}

# --- Dynamic Programmatic Section Maps ---
SECTION_BOOST_MAP = {
    "Pin Description": ["pin", "pinout", "gpio", "gnd", "signal", "terminal"],
    "Pin Configuration": ["pin", "pinout", "gpio", "gnd", "signal", "layout"],
    "Pin Assignments": ["pin", "pinout", "gpio", "gnd", "signal", "layout"],
    "Pin Definitions": ["pin", "pinout", "gpio", "gnd", "definition"],
    "GPIO Matrix": ["gpio", "matrix", "io", "mux", "peripheral"],
    "IO MUX": ["gpio", "mux", "io", "peripheral"],
    "Terminal Functions": ["pin", "terminal", "function", "gnd"],
    "Signal Description": ["signal", "description", "pin", "io"],
    "Absolute Maximum Ratings": ["voltage", "vin", "current", "temperature", "max", "vdd"],
    "Electrical Characteristics": ["voltage", "current", "efficiency", "resistance", "timing", "min", "max"],
    "Table Data": ["table", "typical", "electrical", "specifications", "register"],
    "Overview": ["overview", "features", "description", "architecture"],
    # --- Added engineering sections ---
    "Reference Design": ["reference design", "schematic", "minimum circuit", "application circuit", "decoupling"],
    "Reference Designs": ["reference design", "schematic", "minimum circuit", "application circuit", "decoupling"],
    "Module Schematics": ["schematic", "reference design", "minimum circuit", "pin", "decoupling"],
    "Schematics": ["schematic", "reference design", "minimum circuit", "pin", "decoupling"],
    "Typical Application": ["reference design", "schematic", "minimum circuit", "application circuit", "power"],
    "Typical Application Circuit": ["reference design", "schematic", "minimum circuit", "application circuit", "power"],
    "Typical Applications": ["reference design", "schematic", "minimum circuit", "application circuit", "power"],
    "Application Circuit": ["reference design", "schematic", "minimum circuit", "power"],
    "Application Example": ["reference design", "schematic", "minimum circuit", "power"],
    "PCB Layout": ["pcb", "layout", "hardware design", "land pattern", "footprint"],
    "PCB Layout Guidelines": ["pcb", "layout", "hardware design", "land pattern", "footprint"],
    "Hardware Design": ["hardware design", "pcb", "layout", "schematic", "reference design"],
    "Hardware Design Guidelines": ["hardware design", "pcb", "layout", "schematic", "reference design"],
    "Boot Configuration": ["boot", "strap", "strapping", "startup", "chip_en", "gpio"],
    "Boot Mode": ["boot", "strap", "strapping", "startup", "chip_en"],
    "Strapping Pins": ["boot", "strap", "strapping", "gpio", "chip_en"],
    "Power Consumption": ["power", "current", "consumption", "sleep", "low power"],
    "Current Consumption": ["power", "current", "consumption", "sleep", "low power"],
}

# Dynamic Multiplier: score *= (1 + BASE_BOOST_PER_MATCH * total_matches)
BASE_BOOST_PER_MATCH = 0.05
MAX_SECTION_MULTIPLIER = 1.15  # Protection ceiling capping cumulative growth

# --- API Layer & Crops ---
API_MAX_RETRIES = 4
API_BACKOFF_BASE_SECONDS = 1.5
MAX_CONTEXT_CHARS = 9000

FIGURE_CROP_BAND_HEIGHT = 260     
FIGURE_CROP_PADDING_BELOW = 12    
FIGURE_CROP_SIDE_MARGIN = 20      

KNOWN_MANUFACTURERS = [
    "Texas Instruments", "STMicroelectronics", "Analog Devices", "Microchip",
    "ON Semiconductor", "Infineon", "NXP", "Maxim Integrated", "Diodes Incorporated",
    "Vishay", "ROHM", "Renesas", "Broadcom", "Nordic Semiconductor", "Espressif",
    "Silicon Labs", "Toshiba", "Fairchild Semiconductor", "Bosch", "Honeywell",
]

REVISION_PATTERN = r"\bRev(?:ision)?\b\.?\s*[:\-]?\s*([A-Z]\b|\d+(?:\.\d+)*)"

CATEGORY_KEYWORDS = {
    "Power Supply": ["regulator", "buck", "boost", "converter", "vin", "vout", "switching", "ldo", "power supply", "charger"],
    "Microcontroller": ["microcontroller", "mcu", "cpu", "flash memory", "gpio", "instruction set", "core", "io mux"],
    "Sensor": ["sensor", "temperature", "pressure", "accelerometer", "humidity", "gyroscope", "proximity"],
    "Amplifier": ["operational amplifier", "op-amp", "op amp", "gain", "amplifier"],
    "Communication Interface": ["i2c", "spi", "uart", "can bus", "usb", "rs-485", "rs485"],
    "Memory": ["eeprom", "flash memory", "sram", "memory array"],
    "Passive / Connector": ["relay", "connector", "resistor", "capacitor", "inductor"],
}

KEYWORD_VOCAB = [
    "VIN", "VOUT", "GND", "GPIO", "I2C", "SPI", "UART", "PWM", "ADC", "DAC",
    "current", "voltage", "frequency", "efficiency", "temperature", "resistance",
    "capacitance", "duty cycle", "clock", "interrupt", "power dissipation",
    "switching", "thermal", "package", "pinout", "tx", "rx", "chip_en"
]
MAX_KEYWORDS_PER_CHUNK = 12