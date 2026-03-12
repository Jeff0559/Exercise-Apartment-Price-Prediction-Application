"""
Generates realistic synthetic apartment listing data for the Zurich area.

Run this script once to create data/apartments_data_enriched_with_new_features.csv, which serves
as the input for the full preprocessing and modelling pipeline.

Usage:
    python generate_sample_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
rng = np.random.default_rng(42)

N_SAMPLES = 700
WG_RATIO = 0.05  # share of WG/shared-room listings to inject

# ---------------------------------------------------------------------------
# Zurich-area cities: (name, zip_code, lat, lon, price_multiplier)
# price_multiplier is relative to a base rate of ~26 CHF/sqm/month
# ---------------------------------------------------------------------------
CITY_DATA = [
    ("Zürich", "8001", 47.3766, 8.5423, 1.30),
    ("Zürich", "8002", 47.3594, 8.5362, 1.20),
    ("Zürich", "8003", 47.3680, 8.5218, 1.10),
    ("Zürich", "8004", 47.3751, 8.5280, 1.10),
    ("Zürich", "8005", 47.3830, 8.5205, 1.05),
    ("Zürich", "8006", 47.3887, 8.5430, 1.05),
    ("Zürich", "8008", 47.3600, 8.5571, 1.18),
    ("Zürich", "8032", 47.3634, 8.5484, 1.12),
    ("Zürich", "8037", 47.3946, 8.5282, 0.98),
    ("Zürich", "8045", 47.3667, 8.5087, 0.92),
    ("Zürich", "8046", 47.4270, 8.5145, 0.88),
    ("Zürich", "8047", 47.3862, 8.5005, 0.90),
    ("Zürich", "8048", 47.3837, 8.4850, 0.92),
    ("Zürich", "8049", 47.4010, 8.4950, 0.93),
    ("Zürich", "8051", 47.4026, 8.5715, 0.90),
    ("Zürich", "8057", 47.4091, 8.5468, 1.02),
    ("Zürich", "8064", 47.3862, 8.4940, 0.91),
    ("Dübendorf", "8600", 47.3972, 8.6175, 0.88),
    ("Dübendorf", "8603", 47.3888, 8.6397, 0.85),
    ("Dietikon", "8953", 47.4023, 8.3995, 0.82),
    ("Schlieren", "8952", 47.3960, 8.4479, 0.85),
    ("Kloten", "8302", 47.4508, 8.5848, 0.83),
    ("Regensdorf", "8105", 47.4312, 8.4688, 0.82),
    ("Adliswil", "8134", 47.3108, 8.5238, 0.91),
    ("Wallisellen", "8304", 47.4163, 8.5944, 0.87),
    ("Opfikon", "8152", 47.4280, 8.5720, 0.88),
    ("Uster", "8610", 47.3512, 8.7178, 0.85),
    ("Winterthur", "8400", 47.5002, 8.7298, 0.80),
    ("Winterthur", "8404", 47.4885, 8.7298, 0.78),
    ("Winterthur", "8406", 47.5135, 8.7415, 0.78),
    ("Thalwil", "8800", 47.2920, 8.5661, 0.93),
    ("Horgen", "8810", 47.2582, 8.5950, 0.89),
    ("Küsnacht", "8700", 47.3140, 8.5836, 1.12),
    ("Meilen", "8706", 47.2681, 8.6471, 0.95),
    ("Bülach", "8180", 47.5255, 8.5413, 0.78),
    ("Birmensdorf", "8903", 47.3596, 8.4278, 0.80),
    ("Wädenswil", "8820", 47.2264, 8.6753, 0.85),
    ("Volketswil", "8604", 47.3837, 8.6878, 0.82),
    ("Bassersdorf", "8303", 47.4427, 8.6290, 0.80),
    ("Urdorf", "8902", 47.3836, 8.4228, 0.83),
    ("Pfäffikon ZH", "8330", 47.3716, 8.7816, 0.78),
    ("Wetzikon", "8620", 47.3245, 8.7988, 0.76),
    ("Embrach", "8424", 47.5001, 8.5970, 0.75),
    ("Rümlang", "8153", 47.4470, 8.5355, 0.82),
]

STREETS = [
    "Bahnhofstrasse", "Seestrasse", "Langstrasse", "Birmensdorferstrasse",
    "Rämistrasse", "Forchstrasse", "Schaffhauserstrasse", "Weinbergstrasse",
    "Regensbergstrasse", "Albisriederstrasse", "Militärstrasse", "Quellenstrasse",
    "Aargauerstrasse", "Talstrasse", "Mythenquai", "Bellerivestrasse",
    "Dörflistrasse", "Rosengartenstrasse", "Bucheggstrasse", "Hohlstrasse",
    "Badenerstrasse", "Heinrichstrasse", "Limmatstrasse", "Neptunstrasse",
    "Jupiterstrasse", "Kirchgasse", "Gloriastrasse", "Zürichbergstrasse",
    "Freiestrasse", "Mutschellenstrasse", "Josefstrasse", "Zentralstrasse",
]

PUBLISHERS = [
    "Homegate", "ImmoScout24", "Comparis", "Privat", "Livit AG",
    "Wincasa AG", "Procimmo", "UBS Real Estate", "Helvetia Immobilien",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rooms_label(r: float) -> str:
    """Returns Swiss-style room label: 3 → '3', 3.5 → '3.5'"""
    return str(int(r)) if r == int(r) else f"{int(r)}.5"


def _generate_title(furnished: bool, wg: bool, rooms: float, city: str) -> str:
    apt_type = random.choice(["Wohnung", "Apartment", "Mietwohnung"])
    rl = _rooms_label(rooms)
    if wg:
        return random.choice([
            f"WG-Zimmer in {city}",
            f"Zimmer in WG – {city}",
            f"WG-Zimmer frei – {city}",
            f"Mitbewohner gesucht – {city}",
            f"Shared Room in {city}",
        ])
    if furnished:
        return random.choice([
            f"Möblierte {rl}-Zimmer-{apt_type} in {city}",
            f"Voll möblierte {rl}-Zimmer-{apt_type}",
            f"Möblierte {apt_type} – {city}",
            f"Furnished {rl}-room {apt_type} in {city}",
        ])
    return random.choice([
        f"{rl}-Zimmer-{apt_type} in {city}",
        f"Moderne {rl}-Zimmer-{apt_type} in {city}",
        f"Helle {rl}-Zimmer-{apt_type} – {city}",
        f"Schöne {rl}-Zimmer-{apt_type} in {city}",
        f"Renovierte {rl}-Zimmer-{apt_type} in {city}",
        f"Geräumige {rl}-Zimmer-{apt_type} in {city}",
        f"Zentral gelegene {rl}-Zimmer-{apt_type} – {city}",
    ])


def _generate_description(furnished: bool, parking: bool, wg: bool, city: str) -> str:
    parts: list[str] = []
    if wg:
        parts.append(f"Wir suchen einen neuen Mitbewohner für unsere WG in {city}.")
        parts.append("Das WG-Zimmer ist gemütlich und hell.")
        parts.append("Küche, Bad und Wohnzimmer werden geteilt.")
        parts.append("Ideal für Studierende oder Berufspendler.")
        return " ".join(parts)

    if furnished:
        parts.append("Die Wohnung ist vollständig möbliert und direkt bezugsbereit.")
        if random.random() < 0.6:
            parts.append(
                "Alle nötigen Möbel, Kühlschrank, Geschirrspüler und "
                "Waschmaschine sind vorhanden."
            )
    else:
        parts.append("Gepflegte Wohnung in sehr gutem Zustand.")
        if random.random() < 0.4:
            parts.append("Moderne Einbauküche vorhanden.")

    if parking:
        choice = random.choice(
            ["Garage", "Tiefgaragenplatz", "Parkplatz", "Carport", "Einstellplatz"]
        )
        parts.append(f"{choice} kann dazugemietet werden.")

    extras = [
        "Schöne Aussicht auf die Umgebung.",
        "Südausrichtung mit viel natürlichem Licht.",
        "Ruhige Wohnlage im Quartier.",
        "Gute Anbindung an öffentliche Verkehrsmittel.",
        "Balkon mit schöner Aussicht vorhanden.",
        "Lift im Gebäude vorhanden.",
        "Grosszügiger Keller inklusive.",
        "Ideale Verkehrsanbindung.",
        "Kindergarten und Schule in unmittelbarer Nähe.",
        "Einkaufsmöglichkeiten fussläufig erreichbar.",
        "Frisch renoviert und neu gestrichen.",
        "Modernes Bad mit Dusche und Badewanne.",
        "Ruhige Lage, trotz stadtnaher Position.",
    ]
    parts.extend(random.sample(extras, random.randint(2, 4)))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """Creates a pandas DataFrame with realistic Zurich apartment listings."""
    # Weighted city sampling: Zürich city gets 3× more listings
    city_weights = [3.0 if city == "Zürich" else 1.0 for city, *_ in CITY_DATA]
    total_w = sum(city_weights)
    city_probs = [w / total_w for w in city_weights]

    n_wg = int(n * WG_RATIO)
    n_regular = n - n_wg

    room_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    room_probs  = [0.03, 0.05, 0.12, 0.15, 0.25, 0.18, 0.12, 0.05, 0.03, 0.01, 0.01]

    rows: list[dict] = []

    # ── Regular apartment listings ────────────────────────────────────────────
    for _ in range(n_regular):
        city_idx = rng.choice(len(CITY_DATA), p=city_probs)
        city, zip_code, base_lat, base_lon, price_mult = CITY_DATA[city_idx]

        lat = float(base_lat + rng.normal(0, 0.003))
        lon = float(base_lon + rng.normal(0, 0.004))
        number_of_rooms = float(rng.choice(room_values, p=room_probs))

        size_mean = number_of_rooms * 24 + 10
        apartment_size_sqm = float(
            max(20.0, min(200.0, rng.normal(size_mean, size_mean * 0.15)))
        )
        apartment_size_sqm = round(apartment_size_sqm, 1)

        furnished = bool(rng.random() < 0.15)
        parking   = bool(rng.random() < 0.30)

        base_rate = 26.0  # CHF per sqm per month at baseline
        price = price_mult * (
            base_rate * apartment_size_sqm
            + 180.0 * number_of_rooms
            + (300.0 if furnished else 0.0)
            + (150.0 if parking   else 0.0)
            + float(rng.normal(0, 130))
        )
        price = max(800.0, round(price / 10) * 10)

        # Inject realistic missing values for lat/lon and size/rooms
        lat_val  = round(lat, 6)  if rng.random() > 0.03 else None
        lon_val  = round(lon, 6)  if rng.random() > 0.03 else None
        size_val = round(apartment_size_sqm, 1) if rng.random() > 0.02 else None
        rooms_val = number_of_rooms if rng.random() > 0.02 else None

        street = random.choice(STREETS)
        house_no = random.randint(1, 150)

        rows.append({
            "title":              _generate_title(furnished, False, number_of_rooms, city),
            "description":        _generate_description(furnished, parking, False, city),
            "price":              price,
            "number_of_rooms":    rooms_val,
            "apartment_size_sqm": size_val,
            "address":            f"{street} {house_no}",
            "zip_code":           zip_code,
            "city":               city,
            "latitude":           lat_val,
            "longitude":          lon_val,
            "publisher":          random.choice(PUBLISHERS),
        })

    # ── WG / shared-room listings (to be filtered out later) ─────────────────
    for _ in range(n_wg):
        city_idx = rng.choice(len(CITY_DATA), p=city_probs)
        city, zip_code, base_lat, base_lon, _ = CITY_DATA[city_idx]

        lat = float(base_lat + rng.normal(0, 0.003))
        lon = float(base_lon + rng.normal(0, 0.004))
        wg_price = round(float(rng.uniform(500, 950)) / 10) * 10
        wg_size  = round(float(rng.uniform(12, 25)), 1)
        street   = random.choice(STREETS)

        rows.append({
            "title":              _generate_title(False, True, 1.0, city),
            "description":        _generate_description(False, False, True, city),
            "price":              wg_price,
            "number_of_rooms":    1.0,
            "apartment_size_sqm": wg_size,
            "address":            f"{street} {random.randint(1, 150)}",
            "zip_code":           zip_code,
            "city":               city,
            "latitude":           round(lat, 6),
            "longitude":          round(lon, 6),
            "publisher":          random.choice(PUBLISHERS),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_path = Path("data/apartments_data_enriched_with_new_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(n=N_SAMPLES)
    df.to_csv(output_path, index=False)

    print(f"✓  Generated {len(df)} rows → '{output_path}'")
    print(df[["city", "zip_code", "price", "number_of_rooms", "apartment_size_sqm"]].head(10))
    print("\nPrice statistics:")
    print(df["price"].describe().round(1))
