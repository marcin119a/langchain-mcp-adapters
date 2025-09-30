from mcp.server.fastmcp import FastMCP
import pandas as pd
from utils import load_model, encode_features

mcp = FastMCP("Math")


print("Math MCP server started")
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def search_apartments(min_price: int, max_price: int):
    """Search apartments in Warsaw by price range. Returns max 3 offers with direct URLs to listings."""
    df = pd.read_csv("data/adresowo_warszawa_wroclaw.csv")

    # Czyszczenie cen
    df['price_total_zl'] = (
        df['price_total_zl']
        .astype(str)
        .str.replace(" ", "")
        .str.replace(",", "")
        .str.extract(r"(\d+)")
    )
    df['price_total_zl'] = pd.to_numeric(df['price_total_zl'], errors="coerce")
    results = df[
        (df['city'] == "Warszawa") &
        (df['price_total_zl'] >= min_price) &
        (df['price_total_zl'] <= max_price)
    ].head(3)

    return results[['locality', 'street', 'rooms', 'area_m2', 'price_total_zl', 'url']].to_dict(orient="records")


@mcp.tool()
def predict_price(rooms: int, area_m2: float, locality: str = "Warszawa", 
                  street: str = "unknown", property_type: str = "Mieszkanie", 
                  city: str = "Warszawa") -> float:
    """
    Predict flat price based on features.
    """
    model = load_model()
    df_input = pd.DataFrame([{
        "rooms": rooms,
        "area_m2": area_m2,
        "locality": locality,
        "street": street,
        "property_type": property_type,
        "city": city,
    }])

    df_input = encode_features(df_input)
    features = ['rooms', 'area_m2', 'photos', 'locality', 'street',
                'property_type', 'city', 'owner_direct']
    pred = model.predict(df_input[features])[0]
    return float(pred)


if __name__ == "__main__":
    mcp.run(transport="stdio")