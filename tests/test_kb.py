from optscore.data import kb


def test_drink_attributes_are_known():
    for drink, attrs in kb.DRINKS.items():
        assert attrs <= set(kb.ATTRIBUTES), drink


def test_synonyms_are_unique_across_attributes():
    words = [w for syns in kb.ATTRIBUTES.values() for w in syns]
    assert len(words) == len(set(words))


def test_synonyms_do_not_collide_with_drink_names():
    words = {w for syns in kb.ATTRIBUTES.values() for w in syns}
    assert not words & set(kb.DRINKS)


def test_every_attribute_discriminates():
    for attr in kb.ATTRIBUTES:
        with_attr = [d for d in kb.DRINKS if kb.has_attr(d, attr)]
        assert with_attr and len(with_attr) < len(kb.DRINKS), attr


def test_beer_and_wine_facts_from_the_spec_example():
    assert kb.has_attr("beer", "fizzy")
    assert not kb.has_attr("wine", "fizzy")
    assert kb.attr_of_word("carbonated") == "fizzy"
    assert kb.attr_of_word("nonsense") is None
