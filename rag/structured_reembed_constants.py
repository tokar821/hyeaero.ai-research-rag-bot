"""
Entity types that use structured, entity-based chunking + :mod:`rag.pinecone_metadata`.

``document`` (manuals, PDFs, articles) are **excluded** from the structured refresh — they keep
multi-chunk document chunking.

**Marketplace coverage:** Controller / Aircraft Exchange-style rows typically live in ``aircraft_listings``
(``aircraft_listing``). **AircraftPost** fleet rows use ``aircraftpost_fleet_aircraft`` and are included
in the structured refresh when listed below.

``phlydata_aircraft`` is embedded in Pinecone namespace ``phlydata_aircraft`` (see
:mod:`rag.phlydata_aircraft_embed`); the other types use the **default** namespace (``""``).
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

# Default namespace structured entities (same index as main RAG retrieve).
STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES: Tuple[str, ...] = (
    "aircraft",
    "aircraft_listing",
    "aircraft_sale",
    "faa_registration",
    "aviacost_aircraft_detail",
    "aircraftpost_fleet_aircraft",
)

PHLYDATA_AIRCRAFT_ENTITY_TYPE = "phlydata_aircraft"
PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE = "phlydata_aircraft"

STRUCTURED_AVIATION_ALL_ENTITY_TYPES: FrozenSet[str] = frozenset(
    (*STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES, PHLYDATA_AIRCRAFT_ENTITY_TYPE)
)

DOCUMENT_ENTITY_TYPE = "document"
