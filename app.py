import streamlit as st
from googletrans import Translator
from PIL import Image, ImageDraw
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.applications import EfficientNetB0


@st.cache_resource
def load_models():
    disease_model = tf.keras.models.load_model("models/disease_model.h5", compile=False)
    #type_model = tf.keras.models.load_model("models/type_model.h5", compile=False)
    #size_model = tf.keras.models.load_model("models/size_model.h5", compile=False)
    return disease_model

disease_model = load_models()


def preprocess_image(img, size=224):
    img = img.resize((size, size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img



# ---------- VIDEO HELPER FUNCTION ----------
def center_content(width_ratio=2):
    left, center, right = st.columns([1, width_ratio, 1])
    return center

def show_video(video_path, width=500):
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <video width="{width}" controls>
                <source src="{video_path}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )
def show_video_horizontal(video_path):
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;">
            <video controls style="width:70%; height:auto;">
                <source src="{video_path}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )
    
def show_video_vertical(video_path):
    
    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:center;
            margin: 20px 0;
        ">
            <video controls
                style="
                    width: 320px;
                    aspect-ratio: 9 / 16;
                    object-fit: contain;
                    background: black;
                ">
                <source src="{video_path}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )
    

    
st.markdown("""
<div style="
    background:#f4f6f8;
    padding:35px;
    border-radius:18px;
    margin-bottom:25px;
    color:#111;
    font-size:18px;
">

<h2 style="text-align:center; color:#000000;">
AI‑Powered Prawn Monitoring System for Disease, Type, and Size Prediction
</h2>

<p style="text-align:center; font-size:16px; color:#000000;">
<b>Objective:</b>
To develop an AI‑powered system for identifying prawn disease, type, and size
using image processing and deep learning, helping farmers make timely and
accurate decisions.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background:#f4f6f8;
    padding:35px;
    border-radius:18px;
    color:#111;
    font-size:18px;
">

<p style="text-align:center; color:#000000;">
We are 3rd‑year BTech students at KIETW. We are part of the Artificial Intelligence
Career for Women program, a CSR by Microsoft and SAP, implemented by the Edunet Foundation.
</p>

<div style="display:flex; justify-content:space-between; margin-top:30px;">

<div style="width:55%;">
<p><b>D. Sravanthi</b></p>
<p>N. Hyma Sri</p>
<p>K. Rani</p>
<p>P. Yamini</p>
</div>

<div style="width:35%; text-align:right; color:#000000;">
<p><b>Abdul Aziz Md</b></p>
<p>Master Trainer</p>
<p style="text-align:right; margin-top:-28px; color:#000000;">
Microsoft and SAP, Edunet Foundation
</p>
</div>

</div>

<p style="margin-top:25px; color:#000000;">
<b>Kakinada Institute of Engineering and Technology for Women</b>
</p>



</div>
""", unsafe_allow_html=True)




# ---------------- TRANSLATOR ----------------
translator = Translator()

LANG_CODES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta"
}

def translate_text(text, lang):
    if lang == "English":
        return text
    try:
        return translator.translate(text, dest=LANG_CODES[lang]).text
    except:
        return text  # fallback to English

# ---------------- CSS ----------------
st.markdown("""
<style>
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
lang = "English"

UI_TEXT = {
    "species": "Prawn Species Cultured",
    "white": "White‑leg Prawn",
    "tiger": "Tiger Prawn",
    "scampi": "Scampi Prawn",
    "view": "View",
    "views": "View Prawn Growth",
    "growth": "Prawn Growth",
    "pond": "Pond Preparation",
    "soil": "Soil Management",
    "poly": "Polyculture (Prawns)",
}



POND_PROCESS = {
    "lime": """
Pond lime treatment is an essential step in prawn farming and plays a major role
in improving pond productivity. After drying the pond bottom, agricultural lime
is applied uniformly to correct soil acidity and stabilize pH levels.

Proper liming helps in killing harmful pathogens and improves nutrient availability
in the pond. It also enhances plankton growth, which serves as natural food for prawns.
This process reduces disease risk and improves prawn survival and growth.
""",

    "bottom": """
Pond bottom treatment is carried out after drying and liming.
Organic sludge, leftover feed, and waste materials must be removed properly.
Raking or ploughing the pond bottom improves aeration and prevents toxic gas formation.

This step prepares the pond for fresh water filling and healthy culture conditions.
""",

    "harvest": """
After harvesting the prawns, the pond must be cleaned thoroughly.
Dead organisms and organic waste should be removed completely.
This prevents disease carryover to the next culture cycle.

Proper cleaning improves pond hygiene and ensures better results in the next crop.
""",

    "bleaching": """
Water bleaching treatment is done to disinfect pond water.
Bleaching powder eliminates harmful bacteria, viruses, and unwanted organisms.
It should be applied carefully according to recommended dosage.

This process ensures safe and disease-free water conditions.
""",

    "water": """
After three days of bleaching treatment, water quality parameters such as pH,
dissolved oxygen, and transparency must be checked carefully.
Only after confirming safe conditions should probiotics be applied.

Maintaining good water quality is essential for healthy prawn growth.
"""
}
SOIL_PROCESS = """
Soil management is a crucial factor in successful prawn farming.
The ideal soil pH for prawn culture ranges between 7.5 and 8.5.
Before starting the culture, pond soil should be tested to determine acidity
and nutrient levels.

If the soil is acidic, agricultural lime must be applied to neutralize it.
Proper soil management improves microbial activity and prevents toxic gas
formation such as hydrogen sulfide.

Good soil condition supports healthy plankton growth, maintains water quality,
and creates a stable environment for prawn growth and survival.
"""

POLYCULTURE_TEXT = """
Polyculture in prawn farming involves rearing prawns along with compatible
aquatic species to maintain ecological balance in the pond.

In prawn‑based polyculture systems, proper species selection and stocking
density are important. Polyculture helps in better utilization of pond
resources, improves water quality, and reduces waste accumulation.

This practice enhances overall productivity and minimizes disease risk,
making prawn farming more sustainable and profitable.
"""

PRAWN_INFO = {
    "white_leg": """
Whiteleg shrimp (Litopenaeus vannamei), or Pacific white shrimp, is the world's most cultivated prawn, prized for its fast growth, high-density farming suitability , and tolerance to wide salinity ranges. They are typically long, featuring a mild-sweet flavor and firm texture, making them ideal for global export and culinary use.

Key Details About Whiteleg (Vannamei) Prawns 

1. Habitat & Biology: Native to the Eastern Pacific Ocean, these prawns live in warm tropical waters , with adults inhabiting deeper ocean waters and juveniles inhabiting coastal estuaries.Farming Advantages: They are less aggressive and easier to farm than black tiger prawns, maturing in 100-120 days to a marketable size of 20g+.

3. Production: Global production exceeded 5 million metric tons by 2018, with India being a major producer and exporter, particularly to the U.S..

4. Nutrition: They provide a, tender, firm meat.

5. Health Management: They are often stocked as Specific Pathogen Free (SPF) to manage disease risks in intensive aquaculture systems. 
""",

    "tiger": """
Tiger prawn (Penaeus monodon) is known for its large size and high market value.
It requires good water quality, proper feed management.

Water temperature

Black tiger prawns are a tropical species that achieve the best growth rates when water temperatures are 25–30°C. Lower temperatures will cause feeding to stop at 20°C, and deaths around 14–15°C.

Provided you have controlled stocking rates and selective harvesting in place, this temperature range limits production to:

1.One crop during summer in areas south of Mackay

2.Two crops between Cardwell and Cooktown.
Crops are normally ready for harvest in 120–150 days (when prawns are 25–35g each) however, the time will depend on stocking rates and water temperature.

Harvesting from ponds

Ponds are sometimes partially harvested using traps or seine nets, but more often a drain harvest is used. The water is released through the outlet structure, which has a net fitted over the pipe and the prawns are then caught in this net. Partial harvests may be used early in the season to reduce the density of prawns in the pond and allow the prawns remaining to grow to a larger size.

Water salinity

Black tiger prawns grow best in warm brackish waters (water that is more saline than freshwater, but less than seawater) and can grow quickly under a range of salinities. Prawns can survive in zero salinity (freshwater) for short periods.

Maximum growth rates occur in 15–20 parts per thousand (ppt) salinity—seawater is normally 35ppt.
""",

    "scampi": """
Scampi (specifically referring to the Giant River Prawn, known as Neelakantapu Royya in Telugu, Golda Chingri in Bengali, or Galda Prawn) is a premium freshwater prawn highly regarded for its large size, succulent texture, and sweet, briny flavor.

Here are the key details about Scampi/Neelakantam prawns in India:

Key Characteristics

1. Scientific Name/Type: Macrobrachium rosenbergii (Giant River Prawn).

2. Appearance: They are known for having long, slender claws and a large, fleshy tail.

3. Flavor Profile: Tender texture with a sweet, delicate, and slightly briny taste.

4.Sources: Widely sourced from the rivers of Gujarat and West Bengal (especially the Sundarbans).
 
Pricing and Availability (Kolkata/India Market)

1.Price Range: Typically ranges from ₹500 to ₹800 per kg, though large, premium, or live specimens can cost upwards of ₹900–₹1000 per kg.

2.Sizes: Available in various sizes, including "Medium" (approx. 60-80g per piece) and "Big" (10-14 pieces per kg). 

Culinary Uses

1.Preparation: Ideal for grilling, sautéing, in curries, or served with garlic, butter, and lemon.

2.Popular Dishes: Scampi curry, butter garlic prawns, and grilled scampi. 

Nutritional Benefits

1.Protein-Rich: High in protein and Omega-3 fatty acids.

2.Minerals: A good source of selenium, zinc, phosphorus, and magnesium. """
}

DISEASES = {
    "white_spot": {
        "title": "White Spot Disease",
        "image": "media/white_spot.jpg",
        "info": "Viral disease causing sudden mass mortality. White spots appear on shell.",
        "solution": "Use SPF seed, maintain biosecurity, emergency harvest."
    },
    "black_spot": {
        "title": "Black Spot Disease",
        "image": "media/black_spot.jpeg",
        "info": "Black patches on shell affecting market quality.",
        "solution": "Improve handling, hygiene, and storage."
    },
    "ehp": {
        "title": "EHP Disease",
        "image": "media/ehp.jpg",
        "info": "Parasitic disease causing slow growth and size variation.",
        "solution": "Use EHP-free seed, pond drying, biosecurity."
    },
    "black_gill": {
        "title": "Black Gill Disease",
        "image": "media/black_gill.jpg",
        "info": "Gills turn black due to poor water quality.",
        "solution": "Improve aeration, reduce organic load."
    },
    "yellow_head": {
        "title": "Yellow Head Disease",
        "image": "media/yellow_head.jpg",
        "info": "Highly fatal viral disease with yellowing of head region.",
        "solution": "Use SPF seed and strict biosecurity."
    },
    "tsv": {
        "title": "Taura Syndrome Virus (TSV)",
        "image": "media/tsv.jpg",
        "info": "Viral disease causing weak prawns and reddish tail.",
        "solution": "Use resistant strains and good management."
    },
    "blisters": {
        "title": "Blister Disease",
        "image": "media/blister.jpg",
        "info": "Blisters on shell due to bacterial infection.",
        "solution": "Improve water quality and reduce stress."
    },
    "healthy": {
        "title": "Healthy Prawn",
        "image": "media/healthy.jpg",
        "info": "Clear shell, active movement, uniform growth.",
        "solution": "Maintain good water quality and balanced feed."
    }
}

GROWTH_TEXT = """
Prawn growth occurs through a series of moulting stages where the prawn sheds
its exoskeleton and increases in size.

Growth depends on species, water quality, feed quality, temperature, and pond
management practices.

During early stages, prawns require high‑protein feed and stable dissolved oxygen.
Poor water quality during moulting can cause stress and mortality.

Regular size sampling helps farmers adjust feed quantity and monitor uniform growth.
Good management ensures higher survival and better production.
"""

disease_db = {
    "White Spot Disease": {
        "type": "Viral",
        "risk": "High",
        "description": "Viral disease causing white spots and mass mortality.",
        "recommendation": "Use SPF seed and maintain strict biosecurity."
    },
    "Black Spot Disease": {
        "type": "Bacterial",
        "risk": "Medium",
        "description": "Black spots on shell affecting market quality.",
        "recommendation": "Improve handling, hygiene, and storage."
    },
    "EHP": {
        "type": "Parasitic",
        "risk": "High",
        "description": "Growth retardation and size variation in prawns.",
        "recommendation": "Use EHP‑free seed and pond drying."
    },
    "Black Gill Disease": {
        "type": "Environmental",
        "risk": "Medium",
        "description": "Blackened gills due to poor water quality.",
        "recommendation": "Improve aeration and water exchange."
    },
    "Yellow Head Disease": {
        "type": "Viral",
        "risk": "Very High",
        "description": "Highly fatal disease with yellowing of head.",
        "recommendation": "Use SPF seed and strict biosecurity."
    },
    "Blisters": {
        "type": "Bacterial",
        "risk": "Low",
        "description": "Blisters on shell due to infection.",
        "recommendation": "Reduce stress and improve water quality."
    },
    "TSV": {
        "type": "Viral",
        "risk": "Medium",
        "description": "Weak prawns with reddish tail.",
        "recommendation": "Use resistant strains and good management."
    },
    "Healthy Prawn": {
        "type": "Healthy",
        "risk": "No Risk",
        "description": "Normal color, active movement, uniform growth.",
        "recommendation": "Maintain good water quality and balanced feed."
    }
}

DISEASE_CLASSES = [
    "White Spot Disease",
    "Black Spot Disease",
    "EHP",
    "Black Gill Disease",
    "Yellow Head Disease",
    "Blisters",
    "TSV",
    "Healthy Prawn"
]

TYPE_CLASSES = [
    "White-leg Prawn",
    "Tiger Prawn",
    "Scampi"
]

SIZE_CLASSES = [
    "Very Small",
    "Small",
    "Medium",
    "Large"
]

tabs = st.tabs([
    "🏠 Home",
    "📘 Disease",
    "🔍 Prediction",
    "🎥 Field work",
    "🤖 Chatbot"
    ])



# ---------------- HOME TAB ----------------
# ================= HOME TAB =================
with tabs[0]:


    # ---------- LANGUAGE ----------
    lang = st.selectbox(
        "🌐 Select Language",
        ["English", "Telugu", "Hindi", "Tamil"],
        key="home_lang_select"
    )

    st.markdown("---")

    # ---------- SESSION STATE ----------
    if "active_section" not in st.session_state:
        st.session_state.active_section = None   # pond / soil / poly / growth / feeding

    if "prawn_type" not in st.session_state:
        st.session_state.prawn_type = None       # white_leg / tiger / scampi

    # =====================================================
    # 🔘 MAIN HOME BUTTONS (TOGGLE + AUTO CLOSE)
    # =====================================================
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button(translate_text(UI_TEXT["pond"], lang)):
            st.session_state.active_section = (
                None if st.session_state.active_section == "pond" else "pond"
            )

    with b2:
        if st.button(translate_text(UI_TEXT["soil"], lang)):
            st.session_state.active_section = (
                None if st.session_state.active_section == "soil" else "soil"
            )

    with b3:
        if st.button(translate_text(UI_TEXT["poly"], lang)):
            st.session_state.active_section = (
                None if st.session_state.active_section == "poly" else "poly"
            )

    # =====================================================
    # 🦐 PRAWN SPECIES (ICON ONLY – TOGGLE)
    # =====================================================
    st.markdown(f"## 🦐 {translate_text(UI_TEXT['species'], lang)}")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"### 🦐 {translate_text(UI_TEXT['white'], lang)}")
        if st.button(translate_text(UI_TEXT["view"], lang), key="white_leg"):
            st.session_state.prawn_type = (
                None if st.session_state.prawn_type == "white_leg" else "white_leg"
            )

    with p2:
        st.markdown(f"### 🦐 {translate_text(UI_TEXT['tiger'], lang)}")
        if st.button(translate_text(UI_TEXT["view"], lang), key="tiger"):
            st.session_state.prawn_type = (
                None if st.session_state.prawn_type == "tiger" else "tiger"
            )

    with p3:
        st.markdown(f"### 🦐 {translate_text(UI_TEXT['scampi'], lang)}")
        if st.button(translate_text(UI_TEXT["view"], lang), key="scampi"):
            st.session_state.prawn_type = (
                None if st.session_state.prawn_type == "scampi" else "scampi"
            )

    # ---------- PRAWN DETAILS ----------
    if st.session_state.prawn_type == "white_leg":
        with center_content():
            st.image("media/prawns/whiteleg_prawn.jpg", width=450)
            st.write(translate_text(PRAWN_INFO["white_leg"], lang))

    elif st.session_state.prawn_type == "tiger":
        with center_content():
            st.image("media/prawns/tiger_prawn.jpg", width=450)
            st.write(translate_text(PRAWN_INFO["tiger"], lang))

    elif st.session_state.prawn_type == "scampi":
        with center_content():
            st.image("media/prawns/scampi.jpg", width=450)
            st.write(translate_text(PRAWN_INFO["scampi"], lang))

    # =====================================================
    # 📌 MAIN SECTION CONTENT (ONLY ONE OPEN)
    # =====================================================
    if st.session_state.active_section == "pond":
        st.markdown("## 🧪 Pond Preparation")

        for title, video, text in [
            ("Lime Treatment", "media/pond_lime_treatment.mp4", POND_PROCESS["lime"]),
            ("Bottom Treatment", "media/pond_bottom_treatment.mp4", POND_PROCESS["bottom"]),
            ("Harvest & Cleaning", "media/pond_harvest_after_cleaning.mp4", POND_PROCESS["harvest"]),
            ("Bleaching", "media/bleachingafter_treatment.mp4", POND_PROCESS["bleaching"]),
            ("Water Quality", "media/water_bleaching_treatment.mp4", POND_PROCESS["water"])
        ]:
            st.subheader(title)
            with center_content():
                st.video(video)
                st.write(translate_text(text, lang))

    elif st.session_state.active_section == "soil":
        st.markdown("## 🌱 Soil Management")
        with center_content():
            st.video("media/soil_management.mp4")
            st.write(translate_text(SOIL_PROCESS, lang))

    elif st.session_state.active_section == "poly":
        st.markdown("## 🦐 Polyculture")
        with center_content():
            st.image("media/polyculture.jpg", width=450)
            st.write(translate_text(POLYCULTURE_TEXT, lang))

    # =====================================================
    # 📈 GROWTH (TOGGLE – AUTO CLOSE OTHERS)
    # =====================================================
    st.markdown(f"## 📈 {translate_text(UI_TEXT['growth'], lang)}")

    if st.button(translate_text(UI_TEXT["views"], lang)):
        st.session_state.active_section = (
            None if st.session_state.active_section == "growth" else "growth"
        )

    if st.session_state.active_section == "growth":
        growth_videos = [
            ("1–30 Days", "media/growth_1to30days.mp4"),
            ("26 Days", "media/growth_26day.mp4"),
            ("36 Days", "media/growth_36day.mp4"),
            ("46 Days", "media/growth_43day.mp4"),
            ("53 Days", "media/growth_56day.mp4"),
            ("Harvest Checking", "media/growth_checking.mp4")
        ]

        for title, video in growth_videos:
            st.subheader(title)
            with center_content():
                st.video(video)

        st.write(translate_text(GROWTH_TEXT, lang))
        
        
        
        
# ---------------- DISEASE TAB ----------------
with tabs[1]:

    # -------- LANGUAGE SELECT --------
    lang = st.selectbox(
        "🌐 Select Language",
        ["English", "Telugu", "Hindi", "Tamil"]
    )

    # -------- TITLE --------
    st.markdown(
        f"""
        <h1 style="text-align:center;">
            🦐 {translate_text("Prawn Diseases", lang)}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # -------- DISEASE CARDS --------
    cols = st.columns(4)

    for i, key in enumerate(DISEASES):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    border-radius:12px;
                    background:#1e1e1e;
                    border:1px solid #333;
                    text-align:center;
                ">
                    <h4>{translate_text(DISEASES[key]['title'], lang)}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(translate_text("View", lang), key=key):
    # Toggle logic
                if st.session_state.get("selected_disease") == key:
                    st.session_state.selected_disease = None   # CLOSE
                else:
                    st.session_state.selected_disease = key    # OPEN


    # -------- DISEASE DETAILS --------
    selected = st.session_state.get("selected_disease")

    if selected is not None and selected in DISEASES:
        d = DISEASES[selected]
        
        st.markdown("---")
        st.markdown(f"## 🧬 {translate_text(d['title'], lang)}")
        
        with center_content():
            st.image(d["image"], width=350)

            st.subheader(translate_text("About Disease", lang))
            st.write(translate_text(d["info"], lang))

            st.subheader(translate_text("Prevention & Solution", lang))
            st.write(translate_text(d["solution"], lang))


# ---------------- TAB 3 : PREDICTION ----------------
with tabs[2]:

    st.title("🔍 Automatic Disease Prediction")

    lang = st.selectbox(
        "🌐 Select Language",
        ["English", "Telugu", "Hindi", "Tamil"],
        key="pred_lang"
    )

    uploaded = st.file_uploader(
        "📷 Upload Prawn Image",
        ["jpg", "png", "jpeg"]
    )

    # ✅ Session state
    if "final_prediction" not in st.session_state:
        st.session_state.final_prediction = None

    if uploaded:

        # ---------- LOAD IMAGE ----------
        image = Image.open(uploaded).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size

        # ---------- GREEN BOX ----------
        box_size = int(min(w, h) * 0.6)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=6)

        # ---------- SIZE FROM BOX ----------
        ratio = (box_size * box_size) / (w * h)
        st.write("DEBUG size ratio:", round(ratio, 4))

        if ratio < 0.08:
            prawn_size = "Very Small"
        elif ratio < 0.15:
            prawn_size = "Small"
        elif ratio < 0.22:
            prawn_size = "Medium"
        else:
            prawn_size = "Large"
            
        aspect_ratio = w / h
        
        if aspect_ratio > 1.6:
            prawn_type = "Tiger Prawn",
        elif aspect_ratio > 1.3:
            prawn_type = "White-leg Prawn",
        else:
            prawn_type = "Scampi"

        # ---------- RUN MODELS ONCE ----------
        if st.session_state.final_prediction is None:

            processed = preprocess_image(image)

            disease_pred = disease_model.predict(processed)
            disease_index = int(np.argmax(disease_pred))
            disease_name = DISEASE_CLASSES[disease_index]
            
            disease_index = int(np.argmax(disease_pred))
            disease_name = (
                DISEASE_CLASSES[disease_index]
                if disease_index < len(DISEASE_CLASSES)
                else "Unknown"
                )
            
    

            st.session_state.final_prediction = {
                "disease": disease_name,
                "type": prawn_type,
                "size": prawn_size
            }


        result = st.session_state.final_prediction
        data = disease_db[result["disease"]]

        # ---------- DISPLAY ----------
        with center_content():
            st.image(image, caption="Detected Prawn", width=500)

        st.success(
            f"🦐 {translate_text('Disease Detected', lang)}: "
            f"{translate_text(result['disease'], lang)}"
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### 📏 {translate_text('Size', lang)}")
            st.write(translate_text(result["size"], lang))

            st.markdown(f"### 🦐 {translate_text('Prawn Type', lang)}")
            st.write(translate_text(result["type"], lang))

            st.markdown(f"### ⚠️ {translate_text('Risk Level', lang)}")
            st.write(translate_text(data["risk"], lang))

        with col2:
            st.markdown(f"### 📖 {translate_text('What is the Disease?', lang)}")
            st.write(translate_text(data["description"], lang))

            st.markdown(f"### 🛠️ {translate_text('Solution', lang)}")
            st.write(translate_text(data["recommendation"], lang))
            


# ---------------- TAB 4 ----------------
with tabs[3]:
    st.title("🌾 Prawn Farming – Field Work")

    # All media files (order matters)
    media_files = [
        "assets/field_work/img4.jpeg",
        "assets/field_work/img5.jpeg",
        "assets/field_work/img3.jpeg",
        "assets/field_work/img1.jpeg",
        "assets/field_work/img4.jpeg",
        "assets/field_work/video1.mp4",
        "assets/field_work/video2.mp4",
        "assets/field_work/video3.mp4",
        "assets/field_work/video4.mp4",
    ]

    # Initialize index
    if "fw_index" not in st.session_state:
        st.session_state.fw_index = 0

    col1, col2, col3 = st.columns([1, 6, 1])

    # ◀ Left button
    with col1:
        if st.button("◀"):
            st.session_state.fw_index = (st.session_state.fw_index - 1) % len(media_files)

    # ▶ Right button
    with col3:
        if st.button("▶"):
            st.session_state.fw_index = (st.session_state.fw_index + 1) % len(media_files)

    # Show current media
    current_file = media_files[st.session_state.fw_index]

    with col2:
        if current_file.endswith((".jpg", ".jpeg", ".png")):
            st.image(current_file, width=600, caption="📸 Field Work Image")
        else:
            st.video(current_file)

    st.caption(f"Item {st.session_state.fw_index + 1} of {len(media_files)}")
    
    st.markdown("## 🧾 Field Work Observations")
    st.markdown("""
During the farm survey, farmers shared practical information based on their real farming experience. 
They explained that only a few prawn varieties are commonly preferred in farming because they grow faster, 
adapt well to local environmental conditions, and provide better market value.

Farmers emphasized the importance of proper **pond soil preparation and water management** before starting prawn culture. 
The pond soil must be cleaned, dried, and treated to remove harmful organisms. 
Water quality parameters such as **salinity, pH, and cleanliness** play a major role in prawn health and growth. 
Regular water exchange is required to maintain suitable pond conditions.

For **disease identification**, farmers mainly observe physical symptoms such as changes in prawn color, 
reduced movement, white spots on the body, and slow growth. 
Early identification of disease is very important to prevent its spread.

Farmers also explained that the **cost and quantity of seed stocking** depend on the pond area. 
Based on the size of the farm, a fixed number of prawn seeds are stocked per acre to avoid overcrowding 
and reduce disease risk.

Regarding **minerals and supplements**, farmers apply them at specific growth stages to improve prawn growth, 
strengthen shell formation, and maintain water quality.

In case of disease occurrence, farmers use **disease‑specific medicines**. 
For example, when **White Spot Disease** is observed, immediate action is taken by improving water quality 
and using recommended medicines to control disease spread.

These real‑time field observations helped in understanding the practical challenges faced by farmers 
and supported the design of this project.
""")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # LEFT SIDE – TEAM
    with col1:
        st.markdown("## 👩‍💻 Project Team")
        st.markdown("""
    - **D. Sravanthi**  
    - **N. Hyma Sri**  
    - **K. Rani**  
    - **P. Yamini**
    """)

# RIGHT SIDE – GUIDE
    with col2:
        st.markdown("## 👨‍🏫 Project Guide")
        st.markdown("""
    **Abdul Aziz Md**  
    Master Trainer  
    Edunet Foundation
    """)


# ---------------- TAB 5 ----------------
with tabs[4]:
    q = st.text_input("Ask farmer question")
    if q:
        st.write("🦐 Maintain good water quality and regular monitoring.")
        lang = st.selectbox(
    "🌐 Select Language",
    ["English", "Telugu", "Hindi", "Tamil"],
    key="chat_lang"
)
