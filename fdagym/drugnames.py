"""One generic-name comparison for task generation and hop credit.

A product matches the question's generic when its single active ingredient is
that name, or that name plus salt, ester, or hydrate tokens. A hyphen and a
four-letter FDA biosimilar suffix (natalizumab-sztn) is the same drug.
Combination products, a different drug that only shares a stem (insulin
glargine is not insulin), and a short string inside a longer word do not match.
"""

from __future__ import annotations

import re

# Partial credit needs a real generic. Exact equality still accepts any length.
MIN_PARTIAL_CHARS = 4

# Salt, ester, and hydrate tokens FDA appends to a base name. Includes the
# forms that show up on Drugs@FDA ingredients and label generic names.
_SALT_TOKENS = frozenset(
    {
        "acetate",
        "aluminum",
        "aluminium",
        "anhydrous",
        "besilate",
        "besylate",
        "bicarbonate",
        "bisulfate",
        "bisulphate",
        "bitartrate",
        "bromide",
        "calcium",
        "camsilate",
        "camsylate",
        "carbonate",
        "chloride",
        "citrate",
        "cypionate",
        "decanoate",
        "dihydrochloride",
        "dihydrate",
        "disodium",
        "ditartrate",
        "edisylate",
        "embonate",
        "enacarbil",
        "enanthate",
        "epolamine",
        "esilate",
        "estolate",
        "esylate",
        "fumarate",
        "furoate",
        "gluconate",
        "hci",
        "hcl",
        "hbr",
        "hemihydrate",
        "hemifumarate",
        "hydrate",
        "hyclate",
        "hydchloride",
        "hydrobromide",
        "hydrochloride",
        "hydroiodide",
        "iodide",
        "isethionate",
        "lactate",
        "lauryl",
        "lauroxil",
        "lysine",
        "magnesium",
        "maleate",
        "malate",
        "meglumine",
        "mesilate",
        "mesylate",
        "monohydrate",
        "monosodium",
        "nitrate",
        "oxalate",
        "palmitate",
        "pamoate",
        "pentahydrate",
        "phosphate",
        "potassium",
        "propionate",
        "salicylate",
        "sesquihydrate",
        "sodium",
        "stearate",
        "succinate",
        "sulfate",
        "sulphate",
        "tannate",
        "tartrate",
        "terephthalate",
        "tetrahydrochloride",
        "tosilate",
        "tosylate",
        "trihydrate",
        "trihydrochloride",
        "tromethamine",
        "undecanoate",
        "valerate",
        "xinafoate",
        "zinc",
    }
)

# Same substance, written with a manufacturing word rather than a second drug.
_QUALIFIERS = frozenset({"recombinant", "synthetic"})

# Dosage-form words that trail a salt on some label generic names.
_DOSAGE_TOKENS = frozenset(
    {
        "capsule",
        "capsules",
        "chewable",
        "cream",
        "d/p",
        "d/r",
        "delayed",
        "disintegrating",
        "dr",
        "er",
        "extended",
        "foam",
        "gel",
        "granule",
        "granules",
        "injection",
        "injectable",
        "iv",
        "mcg",
        "mg",
        "ml",
        "odt",
        "oral",
        "orally",
        "patch",
        "release",
        "solution",
        "sr",
        "sublingual",
        "susp",
        "suspension",
        "tablet",
        "tablets",
        "topical",
        "transdermal",
        "xl",
    }
)

_ALLOWED = _SALT_TOKENS | _QUALIFIERS | _DOSAGE_TOKENS
_STRENGTH = re.compile(r"^\d+(?:\.\d+)?(?:mg|mcg|g|ml|%)?$")
_BIOSIMILAR = re.compile(r"^[a-z]{4}$")
_PUNCT = re.compile(r"[-–—_.'()\[\]{}]+")
_COMBO = (" and ", " with ", ";", ",", "+", "&", " / ")


def names_match(raw: str, generic: str) -> bool:
    """True when ``raw`` is ``generic`` or ``generic`` plus salt/ester/hydrate tokens."""
    if not isinstance(raw, str) or not isinstance(generic, str):
        return False
    name = _normalize(raw)
    target = _normalize(generic)
    if not name or not target:
        return False
    if name == target:
        return True
    if any(sep in name for sep in _COMBO):
        return False
    if len(target) < MIN_PARTIAL_CHARS or not name.startswith(target + " "):
        return False
    rest = name[len(target) + 1 :].split()
    hyphenated = any(mark in raw for mark in ("-", "–", "—"))
    return bool(rest) and all(_allowed(token, hyphenated=hyphenated) for token in rest)


def product_matches(product: dict, generic: str) -> bool:
    """True when the product's only active ingredient is ``generic`` or a salt of it."""
    names = ingredient_names(product)
    return len(names) == 1 and names_match(names[0], generic)


def ingredient_names(product: dict) -> list[str]:
    """One entry per ingredient dict. A blank name still counts, so it cannot hide a combo."""
    if not isinstance(product, dict):
        return []
    names: list[str] = []
    for item in product.get("active_ingredients") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        names.append(name.strip() if isinstance(name, str) else "")
    return names


def _normalize(value: str) -> str:
    text = _PUNCT.sub(" ", value.casefold())
    return " ".join(text.split())


def _allowed(token: str, *, hyphenated: bool) -> bool:
    if token in _ALLOWED or _STRENGTH.fullmatch(token) is not None:
        return True
    # FDA biosimilar suffix: reference name, a hyphen, then four letters (natalizumab-sztn).
    return hyphenated and _BIOSIMILAR.fullmatch(token) is not None
