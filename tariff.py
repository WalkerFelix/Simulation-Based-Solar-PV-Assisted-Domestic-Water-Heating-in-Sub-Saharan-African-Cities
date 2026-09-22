# tariff.py

TARIFFS_USD_PER_KWH = {
    "Bloemfontein": 0.186,
    "CapeTown": 0.186,
    "Durban": 0.186,
    "Johannesburg": 0.186,
    "Kimberley": 0.186,
    "Klerksdorp": 0.186,
    "Mbombela": 0.186,
    "Polokwane": 0.186,
    "PortElizabeth": 0.186,
}

def get_city_usd_rate(city: str) -> float:
    """
    Return the flat USD/kWh tariff for the exact city key.
    Raises a clear error if the city is unknown.
    """
    try:
        return float(TARIFFS_USD_PER_KWH[city])
    except KeyError:
        raise KeyError(f"Tariff data does not exist for city: {city}")
