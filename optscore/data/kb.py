"""Toy world: drinks, their attributes, and surface words for each attribute."""

ATTRIBUTES: dict[str, list[str]] = {
    "alcoholic": ["alcoholic", "boozy"],
    "fizzy": ["fizzy", "carbonated", "bubbly", "sparkling"],
    "bitter": ["bitter"],
    "sweet": ["sweet", "sugary"],
    "hot": ["hot", "warm"],
    "sour": ["sour", "tangy"],
    "milky": ["milky", "creamy"],
    "cold": ["cold", "chilled", "icy"],
}

DRINKS: dict[str, frozenset[str]] = {
    "beer": frozenset({"alcoholic", "fizzy", "bitter", "cold"}),
    "wine": frozenset({"alcoholic"}),
    "champagne": frozenset({"alcoholic", "fizzy"}),
    "whiskey": frozenset({"alcoholic", "bitter"}),
    "cider": frozenset({"alcoholic", "fizzy", "sweet", "sour"}),
    "cola": frozenset({"fizzy", "sweet", "cold"}),
    "seltzer": frozenset({"fizzy", "cold"}),
    "kombucha": frozenset({"fizzy", "sour"}),
    "lemonade": frozenset({"sweet", "sour", "cold"}),
    "water": frozenset({"cold"}),
    "juice": frozenset({"sweet", "cold"}),
    "coffee": frozenset({"bitter", "hot"}),
    "tea": frozenset({"hot"}),
    "latte": frozenset({"hot", "milky"}),
    "cocoa": frozenset({"hot", "sweet", "milky"}),
    "milk": frozenset({"milky", "cold"}),
    "milkshake": frozenset({"milky", "sweet", "cold"}),
}


def has_attr(drink: str, attr: str) -> bool:
    return attr in DRINKS[drink]


def attr_of_word(word: str) -> str | None:
    for attr, words in ATTRIBUTES.items():
        if word in words:
            return attr
    return None
