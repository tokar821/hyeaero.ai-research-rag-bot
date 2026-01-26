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


# Registry of extractors
EXTRACTORS = {
    'aircraft_listing': AircraftListingExtractor,
    'document': DocumentExtractor,
    'aircraft': AircraftExtractor,
    'aircraft_sale': AircraftSaleExtractor,
    'faa_registration': FAARegistrationExtractor,
}
