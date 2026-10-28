"""US spellings of British MedDRA preferred terms, emitted as gold aliases.

FAERS reports use MedDRA, which spells the British way (DIARRHOEA, OEDEMA, ANAEMIA). An agent that
answers DIARRHEA has the right answer, so the US spelling goes into gold.aliases.
"""

import re

_RULES = [
    (r"OEA\b", "EA"),  # DIARRHOEA, DYSPNOEA, AMENORRHOEA
    (r"\bOEDEM", "EDEM"),  # OEDEMA, OEDEMA PERIPHERAL
    (r"OEDEM", "EDEM"),  # ANGIOOEDEMA
    (r"OESOPHAG", "ESOPHAG"),
    (r"\bOESTR", "ESTR"),
    (r"\bFOET", "FET"),
    (r"AEMIA", "EMIA"),  # ANAEMIA, HYPOKALAEMIA, SEPTICAEMIA, LEUKAEMIA
    (r"HAEM", "HEM"),  # HAEMORRHAGE, HAEMATOMA, HAEMOGLOBIN
    (r"AEMI", "EMI"),  # ISCHAEMIC
    (r"PAED", "PED"),
    (r"GYNAEC", "GYNEC"),
    (r"CAESAR", "CESAR"),
    (r"\bFAEC", "FEC"),
    (r"ANAESTH", "ANESTH"),
    (r"ISATION\b", "IZATION"),  # HOSPITALISATION
    (r"ISED\b", "IZED"),  # NORMALISED
    (r"\b(TUM|BEHAVI|COL|HUM|OD|VAP|LAB)OUR", r"\1OR"),
]


def us_spelling(term: str) -> str:
    out = term.upper()
    for pattern, repl in _RULES:
        out = re.sub(pattern, repl, out)
    return out


def aliases(term: str) -> list[str]:
    us = us_spelling(term)
    return [us] if us != term.upper() else []
