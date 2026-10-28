from fdagym.tasks.meddra import aliases, us_spelling


def test_british_spellings_gain_a_us_alias():
    assert aliases("DIARRHOEA") == ["DIARRHEA"]
    assert us_spelling("OEDEMA PERIPHERAL") == "EDEMA PERIPHERAL"
    assert aliases("ANAEMIA") == ["ANEMIA"]
    assert aliases("HAEMORRHAGE") == ["HEMORRHAGE"]
    assert aliases("INTERNATIONAL NORMALISED RATIO INCREASED") == ["INTERNATIONAL NORMALIZED RATIO INCREASED"]


def test_terms_without_a_spelling_difference_have_no_alias():
    assert aliases("NAUSEA") == []
    assert aliases("FATIGUE") == []
