# Extract premium trims, sport packages, or body types
df['is_sport'] = df['model'].str.contains('Sport|S-Line|M Sport', case=False).astype(int)
df['is_premium'] = df['model'].str.contains('Premium|Platinum|Limited', case=False).astype(int)
df['is_4WD_AWD'] = df['model'].str.contains('4WD|xDrive|Quattro|AWD', case=False).astype(int)
df.drop(['model'], axis=1, inplace=True)

def clean_and_map_interior_colors_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['int_col', 'ext_col']:
        df = df.copy()

        # Step 1: Vectorized String Normalization
        # Lowercase, strip whitespace, and eliminate junk characters across the entire Series
        clean_series = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"[.\-]", "", regex=True)
        )

        # Step 2: Vectorized Handling of Corrupted Token (ΓÇô), Empty Strings, or Literal NaNs
        # This turns them into a single string 'unknown' to make regex matching safe
        junk_mask = clean_series.isin(["γçô", "nan", "other", ""]) | clean_series.isna()
        clean_series = np.where(junk_mask, "unknown", clean_series)
        clean_series = pd.Series(clean_series)

        # Step 3: Define Categorization Rules using Vectorized Regex Patterns
        # '|'.join() creates an "OR" regex matching configuration (e.g., 'black|blk|ebony')
        conditions = [
            clean_series.str.contains(r"black|blk|ebony|nero|charcoal|obsidian|beluga|amg|graphite|carbon"),
            clean_series.str.contains(r"beige|tan|parchment|sandstone|canberra|shara|macchiato|almond|shale|cashmere|linen|ivory|silk"),
            clean_series.str.contains(r"gray|grey|slate|pewter|titan|boulder|ash|platinum|silver|galvanized|gideon"),
            clean_series.str.contains(r"brown|walnut|espresso|caramel|cappuccino|nougat|sarder|mesa|tupelo|mocha|saddle|auburn|amber|brandy|mountain|aragon|chestnut|cocoa|dune|roast"),
            clean_series.str.contains(r"red|hotspur|rioja|pimento|magma|garnet|chateau|adrenaline"),
            clean_series.str.contains(r"blue|navy|cobalt|rhapsody|charles|mistral|porpoise"),
            clean_series.str.contains(r"orange|sakhir|kyalami|giallo|taurus|yellow"),
            clean_series.str.contains(r"white|ice|pearl|grace|cloud|whisper|bianco|polar"),
            clean_series == "unknown"
        ]

        # Matching outputs for each condition group above
        choices = [
            "Black", 
            "Beige", 
            "Gray", 
            "Brown", 
            "Red", 
            "Blue", 
            "Orange", 
            "White", 
            "Unknown"
        ]

        # Step 4: Execute np.select Vectorized Mapping Engine
        # Any value that fails to match the rules above defaults to "Other"
        df[col] = np.select(conditions, choices, default="Other")
    return df


    df = clean_and_map_interior_colors_vectorized(df)

    mode_accident = df['accident'].mode()
accident_map = {
    'no accidents': 0,
    'At least 1 accident or damage reported': 1}
df['accident'] = df['accident'].map(accident_map)
df['accident'] = df['accident'].fillna(0)


def clean_and_feature_engineer_transmission(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step 1: Broad Normalization (Lowercase and clean whitespace)
    raw_series = df["transmission"].astype(str).str.lower().str.strip()

    # Create masks for clean handling of obvious noise tokens
    junk_strings = ["γçô", "scheduled for or in production", "nan", ""]
    raw_series = np.where(raw_series.isin(junk_strings), "unknown", raw_series)
    raw_series = pd.Series(raw_series)  # Ensure it's a Series for regex operations
    # ------------------------------------------------------------------ #
    # Feature 1: Vectorized Transmission Type Extraction                 #
    # ------------------------------------------------------------------ #
    # Define boolean arrays based on keyword hierarchies

    is_cvt = raw_series.str.contains(r"cvt|variable")
    is_manual = raw_series.str.contains(r"m/t|manual|mt")
    
    # "Automatic" checks for explicit auto terms OR general shorthand like 'a/t' or 'at'
    is_auto = raw_series.str.contains(r"a/t|automatic|auto|at|pdk|dct|steptronic|tronic|shift")

    # Combine conditions using NumPy select (Order matters: CVT checked first)
    type_conditions = [is_cvt, is_manual, is_auto, raw_series == "unknown"]
    type_choices = ["CVT", "Manual", "Automatic", "Unknown"]
    
    df["transmission_type"] = np.select(type_conditions, type_choices, default="Automatic")

    # ------------------------------------------------------------------ #
    # Feature 2: Vectorized Gear/Speed Count Extraction                   #
    # ------------------------------------------------------------------ #
    # Use regular expressions to extract any numbers sitting right next to "speed", "spd", or "speed"
    # Example: "8-speed a/t" -> "8", "automatic, 9-spd" -> "9"
    raw_series = pd.Series(raw_series)  # Ensure it's a Series for regex operations

    extracted_gears = raw_series.str.extract(r"(\d+)\s*(?:-speed|spd|speed|gear)")
    
    # Fallback regex check: just look for standalone numbers if the speed suffix wasn't used
    fallback_gears = raw_series.str.extract(r"\b(\d+)\b")
    
    # Fill missing values from the first regex pass with matches from the fallback pass
    final_gears = extracted_gears[0].fillna(fallback_gears[0])
    
    # Convert to numeric float so it can handle missing values cleanly as NaNs
    df["transmission_gears"] = pd.to_numeric(final_gears, errors="coerce")

    # Business rule correction: If it's a CVT, it shouldn't have discrete gears.
    # We set CVT gear counts to 0 or leave as NaN based on model preference.
    df.loc[df["transmission_type"] == "CVT", "transmission_gears"] = 0
    
    # Fill remaining completely unidentifiable gear numbers (like string "f" or "automatic") with median
    # (Do this processing step within your pipeline imputer setup later if preferred)
    df.drop(['transmission'], axis=1, inplace=True)
    return df




    df = clean_and_feature_engineer_transmission(df)
df.head()


import re
def process_engine_specs_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step 1: Base Normalization and Encoding Cleanup
    raw_series = df["engine"].astype(str).str.strip()

    # Detect corrupted placeholder strings and fill with a safe indicator
    junk_mask = raw_series.isin(["ΓÇô", "nan", ""]) | raw_series.isna()
    raw_series = np.where(junk_mask, "unknown", raw_series)
    raw_series = pd.Series(raw_series)  # Ensure it's a Series for regex operations

    # ------------------------------------------------------------------
    # Feature 1: Vectorized Horsepower (HP) Extraction
    # ------------------------------------------------------------------
    # Extract any numeric sequences preceding 'HP' or 'hp'
    df["engine_hp"] = pd.to_numeric(
        raw_series.str.extract(r"(\d+(?:\.\d+)?)\s*HP", flags=re.IGNORECASE)[0],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # Feature 2: Vectorized Displacement (Liters) Extraction
    # ------------------------------------------------------------------
    # Target formats: "2.0L", "3.0 Liter", "4L"
    disp_extracted = raw_series.str.extract(
        r"(\d+(?:\.\d+)?)\s*(?:L|Liter)", flags=re.IGNORECASE
    )[0]
    df["engine_displacement"] = pd.to_numeric(disp_extracted, errors="coerce")

    # ------------------------------------------------------------------
    # Feature 3: Vectorized Cylinder Count Extraction
    # ------------------------------------------------------------------
    # Look for explicitly stated names: "8 Cylinder", "10 Cylinder"
    cyl_word_match = raw_series.str.extract(
        r"(\d+)\s*Cylinder", flags=re.IGNORECASE
    )[0]

    # Look for structural layout shorthand expressions: "V6", "I4", "H6", "V-10"
    cyl_short_match = raw_series.str.extract(
        r"\b(?:V|I|H|V-)(\d+)\b", flags=re.IGNORECASE
    )[0]

    # Look for specialized text expressions: "Straight 6", "Flat 6"
    cyl_text_match = np.where(
        raw_series.str.contains(r"Straight 6|Flat 6", flags=re.IGNORECASE),
        "6",
        None,
    )

    # Coalesce the extractions prioritising explicit words, then shorthand, then custom text
    final_cylinders = (
        cyl_word_match.fillna(cyl_short_match)
        .fillna(pd.Series(cyl_text_match, index=df.index))
    )
    df["engine_cylinders"] = pd.to_numeric(final_cylinders, errors="coerce")

    # ------------------------------------------------------------------
    # Feature 4: Vectorized Fuel/Powertrain Category Assignment
    # ------------------------------------------------------------------
    # Generate clean boolean evaluation structures across the Series
    is_electric = raw_series.str.contains(
        r"Electric Motor|Electric Fuel|Electric", flags=re.IGNORECASE
    )
    is_hybrid = raw_series.str.contains(r"Hybrid|Plug-In", flags=re.IGNORECASE)
    is_diesel = raw_series.str.contains(r"Diesel", flags=re.IGNORECASE)
    is_flex = raw_series.str.contains(r"Flex Fuel|Flexible", flags=re.IGNORECASE)
    is_gasoline = raw_series.str.contains(
        r"Gasoline|Gas|GDI|MPFI", flags=re.IGNORECASE
    )

    # Map conditions onto target categories in priority sequence
    fuel_conditions = [
        is_electric & ~is_hybrid,  # Pure Electric
        is_hybrid,
        is_diesel,
        is_flex,
        is_gasoline,
        raw_series == "unknown",
    ]
    fuel_choices = [
        "Electric",
        "Hybrid",
        "Diesel",
        "Flex Fuel",
        "Gasoline",
        "Unknown",
    ]

    df["fuel_type_engine"] = np.select(fuel_conditions, fuel_choices, default=None)

    # ------------------------------------------------------------------
    # Step 5: Post-Extraction Adjustments for Electric Vehicles
    # ------------------------------------------------------------------
    # Pure Electric vehicles do not have displacement sizes or internal combustion cylinders
    df.loc[df["fuel_type_engine"] == "Electric", ["engine_displacement", "engine_cylinders"]] = 0
    df.drop(['engine'], axis=1, inplace=True)
    return df


    df = process_engine_specs_vectorized(df) 
df.head()


