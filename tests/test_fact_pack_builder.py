from services.broker_execution.fact_pack_builder import (
    attach_fact_pack_to_data_used,
    build_fact_pack,
    render_fact_pack_for_llm_context,
)


def test_build_fact_pack_tail_facts():
    du = {
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "Acme LLC"},
        ],
        "tail_registration": "N807JS",
    }
    pack = build_fact_pack("Who owns N807JS?", du)
    assert pack["facts"]
    block = render_fact_pack_for_llm_context(pack)
    assert "VERIFIED FACT PACK" in block
    assert "Acme" in block


def test_attach_fact_pack_observability():
    du = {"tail_facts": [{"kind": "ownership", "label": "Owner", "value": "Test"}]}
    block = attach_fact_pack_to_data_used("Who owns N123AB?", du)
    assert du.get("fact_pack_fact_count") >= 1
    assert block
