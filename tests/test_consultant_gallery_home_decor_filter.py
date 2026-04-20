from services.consultant_aircraft_images import _non_aviation_interior_spam_row


def test_tollbrothers_natural_interior_is_spam():
    row = {
        "url": "https://www.tollbrothers.com/blog/natural-interior-design-ideas",
        "description": "Natural interior design style guide",
        "page_url": "",
    }
    assert _non_aviation_interior_spam_row(row) is True


def test_oem_challenger_not_spam():
    row = {
        "url": "https://bombardier.com/en/magazine/flight/inside-challenger-350",
        "description": "Inside the Challenger 350",
        "page_url": "",
    }
    assert _non_aviation_interior_spam_row(row) is False
