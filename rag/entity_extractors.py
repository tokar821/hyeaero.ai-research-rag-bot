"""Entity extractors for different data types.

Converts PostgreSQL records into text suitable for embedding.
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Base class for entity extractors."""
    
    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from record for embedding.
        
        Args:
            record: Database record as dictionary
            
        Returns:
            Text string or None if no extractable text
        """
        raise NotImplementedError
    
    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for the record.
        
        Args:
            record: Database record as dictionary
            
        Returns:
            Metadata dictionary
        """
        raise NotImplementedError


class AircraftListingExtractor(EntityExtractor):
    """Extractor for aircraft_listings table."""
    
    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from aircraft listing."""
        parts = []
        
        # Basic info
        if record.get('manufacturer') or record.get('model'):
            parts.append(f"Aircraft: {record.get('manufacturer', '')} {record.get('model', '')}".strip())
        
        if record.get('manufacturer_year'):
            parts.append(f"Year: {record.get('manufacturer_year')}")
        
        if record.get('category'):
            parts.append(f"Category: {record.get('category')}")
        
        # Pricing
        if record.get('ask_price'):
            parts.append(f"Asking Price: ${record.get('ask_price'):,.0f}")
        if record.get('sold_price'):
            parts.append(f"Sold Price: ${record.get('sold_price'):,.0f}")
        
        # Location
        if record.get('location'):
            parts.append(f"Location: {record.get('location')}")
        if record.get('based_at'):
            parts.append(f"Based at: {record.get('based_at')}")
        
        # Description
        if record.get('description'):
            parts.append(f"Description: {record.get('description')}")
        
        # Features
        if record.get('features'):
            features = record.get('features')
            if isinstance(features, str):
                try:
                    features = json.loads(features)
                except:
                    features = [features]
            if isinstance(features, list):
                parts.append(f"Features: {', '.join(str(f) for f in features)}")
        
        # Avionics
        if record.get('avionics_description'):
            parts.append(f"Avionics: {record.get('avionics_description')}")
        if record.get('avionics_list'):
            parts.append(f"Avionics List: {record.get('avionics_list')}")
        
        # Airframe
        if record.get('airframe_total_time'):
            parts.append(f"Airframe Total Time: {record.get('airframe_total_time')} hours")
        if record.get('airframe_total_cycles'):
            parts.append(f"Airframe Cycles: {record.get('airframe_total_cycles')}")
        
        # Programs
        if record.get('engine_program'):
            parts.append(f"Engine Program: {record.get('engine_program')}")
        if record.get('apu_program'):
            parts.append(f"APU Program: {record.get('apu_program')}")
        
        # Status
        if record.get('listing_status'):
            parts.append(f"Status: {record.get('listing_status')}")
        
        if not parts:
            return None
        
        return "\n".join(parts)
    
    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for aircraft listing."""
        return {
            'entity_type': 'aircraft_listing',
            'entity_id': str(record.get('id')),
            'source_platform': record.get('source_platform'),
            'listing_url': record.get('listing_url'),
            'manufacturer': record.get('manufacturer'),
            'model': record.get('model'),
            'year': record.get('manufacturer_year'),
            'listing_status': record.get('listing_status'),
            'ingestion_date': str(record.get('ingestion_date')) if record.get('ingestion_date') else None,
        }


class DocumentExtractor(EntityExtractor):
    """Extractor for documents table."""
    
    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from document."""
        return record.get('extracted_text')
    
    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for document."""
        return {
            'entity_type': 'document',
            'entity_id': str(record.get('id')),
            'document_id': str(record.get('id')),  # For backward compatibility
            'source_platform': record.get('source_platform'),
            'document_type': record.get('document_type'),
            'file_name': record.get('file_name'),
            'ingestion_date': str(record.get('ingestion_date')) if record.get('ingestion_date') else None,
        }


class AircraftExtractor(EntityExtractor):
    """Extractor for aircraft master table."""
    
    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from aircraft record."""
        parts = []
        
        if record.get('manufacturer') or record.get('model'):
            parts.append(f"Aircraft: {record.get('manufacturer', '')} {record.get('model', '')}".strip())
        
        if record.get('serial_number'):
            parts.append(f"Serial Number: {record.get('serial_number')}")
        
        if record.get('registration_number'):
            parts.append(f"Registration: {record.get('registration_number')}")
        
        if record.get('manufacturer_year'):
            parts.append(f"Year: {record.get('manufacturer_year')}")
        
        if record.get('category'):
            parts.append(f"Category: {record.get('category')}")
        
        if record.get('aircraft_status'):
            parts.append(f"Status: {record.get('aircraft_status')}")
        
        if record.get('condition'):
            parts.append(f"Condition: {record.get('condition')}")
        
        if record.get('registration_country'):
            parts.append(f"Registration Country: {record.get('registration_country')}")
        
        if record.get('based_country'):
            parts.append(f"Based Country: {record.get('based_country')}")
        
        if record.get('type_aircraft'):
            parts.append(f"Type: {record.get('type_aircraft')}")
        
        if record.get('type_engine'):
            parts.append(f"Engine Type: {record.get('type_engine')}")
        
        if not parts:
            return None
        
        return "\n".join(parts)
    
    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for aircraft."""
        return {
            'entity_type': 'aircraft',
            'entity_id': str(record.get('id')),
            'serial_number': record.get('serial_number'),
            'registration_number': record.get('registration_number'),
            'manufacturer': record.get('manufacturer'),
            'model': record.get('model'),
            'year': record.get('manufacturer_year'),
        }


class AircraftSaleExtractor(EntityExtractor):
    """Extractor for aircraft_sales table."""
    
    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from aircraft sale."""
        parts = []
        
        if record.get('manufacturer') or record.get('model'):
            parts.append(f"Aircraft Sale: {record.get('manufacturer', '')} {record.get('model', '')}".strip())
        
        if record.get('serial_number'):
            parts.append(f"Serial Number: {record.get('serial_number')}")
        
        if record.get('date_sold'):
            parts.append(f"Date Sold: {record.get('date_sold')}")
        
        if record.get('sold_price'):
            parts.append(f"Sold Price: ${record.get('sold_price'):,.0f}")
        
        if record.get('transaction_status'):
            parts.append(f"Transaction Status: {record.get('transaction_status')}")
        
        if record.get('days_on_market'):
            parts.append(f"Days on Market: {record.get('days_on_market')}")
        
        if record.get('seller'):
            parts.append(f"Seller: {record.get('seller')}")
        
        if record.get('buyer'):
            parts.append(f"Buyer: {record.get('buyer')}")
        
        if record.get('features'):
            features = record.get('features')
            if isinstance(features, str):
                try:
                    features = json.loads(features)
                except:
                    features = [features]
            if isinstance(features, list):
                parts.append(f"Features: {', '.join(str(f) for f in features)}")
        
        if not parts:
            return None
        
        return "\n".join(parts)
    
    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for aircraft sale."""
        return {
            'entity_type': 'aircraft_sale',
            'entity_id': str(record.get('id')),
            'serial_number': record.get('serial_number'),
            'date_sold': str(record.get('date_sold')) if record.get('date_sold') else None,
            'transaction_status': record.get('transaction_status'),
            'source_platform': record.get('source_platform'),
        }


class FAARegistrationExtractor(EntityExtractor):
    """Extractor for faa_registrations table."""
    
    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from FAA registration."""
        parts = []
        
        if record.get('n_number'):
            parts.append(f"FAA Registration: {record.get('n_number')}")
        
        if record.get('serial_number'):
            parts.append(f"Serial Number: {record.get('serial_number')}")
        
        if record.get('registrant_name'):
            parts.append(f"Registrant: {record.get('registrant_name')}")
        
        if record.get('city') or record.get('state'):
            location = f"{record.get('city', '')}, {record.get('state', '')}".strip(', ')
            if location:
                parts.append(f"Location: {location}")
        
        if record.get('certification'):
            parts.append(f"Certification: {record.get('certification')}")
        
        if record.get('type_aircraft'):
            parts.append(f"Aircraft Type: {record.get('type_aircraft')}")
        
        if record.get('type_engine'):
            parts.append(f"Engine Type: {record.get('type_engine')}")
        
        if record.get('cert_issue_date'):
            parts.append(f"Certificate Issue Date: {record.get('cert_issue_date')}")
        
        if not parts:
            return None
        
        return "\n".join(parts)
    
    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for FAA registration."""
        return {
            'entity_type': 'faa_registration',
            'entity_id': str(record.get('id')),
            'n_number': record.get('n_number'),
            'serial_number': record.get('serial_number'),
            'ingestion_date': str(record.get('ingestion_date')) if record.get('ingestion_date') else None,
        }


class AviacostAircraftDetailExtractor(EntityExtractor):
    """Extractor for aviacost_aircraft_details table (operating cost & specs by aircraft type)."""

    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        """Extract text from Aviacost aircraft detail for RAG."""
        parts = []

        if record.get("name"):
            parts.append(f"Aircraft type: {record['name']}")
        if record.get("manufacturer_name"):
            parts.append(f"Manufacturer: {record['manufacturer_name']}")
        if record.get("category_name"):
            parts.append(f"Category: {record['category_name']}")
        if record.get("description"):
            parts.append(record["description"])
        if record.get("avionics"):
            parts.append(f"Avionics: {record['avionics']}")
        if record.get("years_in_production"):
            parts.append(f"Years in production: {record['years_in_production']}")
        if record.get("average_pre_owned_price") is not None:
            parts.append(f"Average pre-owned price: ${float(record['average_pre_owned_price']):,.0f}")
        if record.get("variable_cost_per_hour") is not None:
            parts.append(f"Variable cost per hour: ${float(record['variable_cost_per_hour']):,.2f}")
        if record.get("fuel_gallons_per_hour") is not None:
            parts.append(f"Fuel: {record['fuel_gallons_per_hour']} gal/hr")
        if record.get("normal_cruise_speed_kts") is not None:
            parts.append(f"Normal cruise: {record['normal_cruise_speed_kts']} kts")
        if record.get("seats_full_range_nm") is not None:
            parts.append(f"Range: {record['seats_full_range_nm']} nm (seats full)")
        if record.get("typical_passenger_capacity_max") is not None:
            parts.append(f"Max passengers: {record['typical_passenger_capacity_max']}")
        if record.get("powerplant"):
            parts.append(f"Powerplant: {record['powerplant']}")
        if record.get("engine_model"):
            parts.append(f"Engine model: {record['engine_model']}")

        if not parts:
            return None
        return "\n".join(parts)

    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for Aviacost aircraft detail."""
        return {
            "entity_type": "aviacost_aircraft_detail",
            "entity_id": str(record.get("id")),
            "source_platform": "aviacost",
            "name": record.get("name"),
            "manufacturer_name": record.get("manufacturer_name"),
            "category_name": record.get("category_name"),
            "aircraft_detail_id": record.get("aircraft_detail_id"),
            "ingestion_date": str(record.get("ingestion_date")) if record.get("ingestion_date") else None,
        }


class AircraftPostFleetAircraftExtractor(EntityExtractor):
    """Extractor for aircraftpost_fleet_aircraft table (fleet stats by serial)."""

    @staticmethod
    def extract_text(record: Dict[str, Any]) -> Optional[str]:
        parts: List[str] = []

        if record.get("make_model_name"):
            parts.append(f"Make/Model: {record.get('make_model_name')}")
        if record.get("serial_number"):
            parts.append(f"Serial Number: {record.get('serial_number')}")
        if record.get("registration_number"):
            parts.append(f"Registration: {record.get('registration_number')}")

        if record.get("mfr_year") is not None:
            parts.append(f"MFR Year: {record.get('mfr_year')}")
        if record.get("eis_date"):
            parts.append(f"EIS Date: {record.get('eis_date')}")
        if record.get("country_code"):
            parts.append(f"Country: {record.get('country_code')}")
        if record.get("base_code"):
            parts.append(f"Base: {record.get('base_code')}")

        if record.get("airframe_hours") is not None:
            parts.append(f"Airframe Hours: {record.get('airframe_hours')}")
        if record.get("total_landings") is not None:
            parts.append(f"Total Landings: {record.get('total_landings')}")
        if record.get("prior_owners") is not None:
            parts.append(f"Prior Owners: {record.get('prior_owners')}")
        if record.get("for_sale") is True:
            parts.append("For Sale: Yes")
        elif record.get("for_sale") is False:
            parts.append("For Sale: No")
        if record.get("passengers") is not None:
            parts.append(f"Passengers: {record.get('passengers')}")

        if record.get("engine_program_type"):
            parts.append(f"Engine Program Type: {record.get('engine_program_type')}")
        if record.get("apu_program"):
            parts.append(f"APU Program: {record.get('apu_program')}")
        if record.get("owner_url"):
            parts.append(f"Owner URL: {record.get('owner_url')}")

        # Include a compact list of equipment section names for retrieval (avoid huge text)
        sections = record.get("sections")
        if isinstance(sections, str):
            try:
                sections = json.loads(sections)
            except Exception:
                sections = None
        if isinstance(sections, dict) and sections:
            section_names = [str(k) for k in sections.keys()][:12]
            parts.append(f"Equipment sections: {', '.join(section_names)}")

        if not parts:
            return None
        return "\n".join(parts)

    @staticmethod
    def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_type": "aircraftpost_fleet_aircraft",
            "entity_id": str(record.get("id")),
            "source_platform": "aircraftpost",
            "make_model_id": record.get("make_model_id"),
            "make_model_name": record.get("make_model_name"),
            "aircraft_entity_id": record.get("aircraft_entity_id"),
            "serial_number": record.get("serial_number"),
            "registration_number": record.get("registration_number"),
            "ingestion_date": str(record.get("ingestion_date")) if record.get("ingestion_date") else None,
        }


# Registry of extractors
EXTRACTORS = {
    "aircraft_listing": AircraftListingExtractor,
    "document": DocumentExtractor,
    "aircraft": AircraftExtractor,
    "aircraft_sale": AircraftSaleExtractor,
    "faa_registration": FAARegistrationExtractor,
    "aviacost_aircraft_detail": AviacostAircraftDetailExtractor,
    "aircraftpost_fleet_aircraft": AircraftPostFleetAircraftExtractor,
}
