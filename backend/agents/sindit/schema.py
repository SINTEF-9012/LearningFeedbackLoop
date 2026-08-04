"""
Neo4j Graph Schema for Machine Asset Management.

Implements a machine-centric asset graph with status as first-class nodes.

Core node model:
  - Machine: unique asset with metadata (machine_id, name, type, vendor, location, active)
  - Status: current operating state (code, severity, label)
  - Description: human-readable asset info (text, lang, source)
  - StatusEvent: historical state transitions (timestamp, from_status, to_status, reason, source)

Key relationships:
  - (Machine)-[:CURRENT_STATUS]->(Status)
  - (Machine)-[:HAS_DESCRIPTION]->(Description)
  - (Machine)-[:HAS_STATUS_EVENT]->(StatusEvent)
  - (ProductionLine)-[:HAS_MACHINE]->(Machine)

This schema prioritizes:
  1. Quick status queries: (Machine)-[:CURRENT_STATUS] for active status
  2. History traversal: (Machine)-[:HAS_STATUS_EVENT] ordered by timestamp
  3. Asset metadata: (Machine)-[:HAS_DESCRIPTION] for documentation
  4. Organization: ProductionLine → Machines (grouping/hierarchy)
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


class StatusCode(str, Enum):
    """Machine operating state codes."""
    ACTIVE = "ACTIVE"           # Running, cutting
    IDLE = "IDLE"               # At rest, powered on
    STOPPED = "STOPPED"         # Powered off or emergency stop
    MAINTENANCE = "MAINTENANCE" # Scheduled unavailable
    FAULT = "FAULT"             # Error, alarm, or degraded mode


class StatusSeverity(str, Enum):
    """Severity level for faults / warnings."""
    INFO = "INFO"           # Informational
    WARNING = "WARNING"     # Non-critical alert
    CRITICAL = "CRITICAL"  # Requires immediate attention


@dataclass
class Machine:
    """Machine asset node."""
    machine_id: str                    # Unique identifier (e.g., MACHINE_A1)
    name: str                          # Display name
    type: str                          # Equipment type (e.g., "5-axis milling center")
    vendor: Optional[str] = None       # Manufacturer
    location: Optional[str] = None     # Physical location / line
    active: bool = True                # Is this machine in service?
    updated_at: Optional[datetime] = None  # Last update timestamp
    
    def to_neo4j_properties(self) -> Dict[str, any]:
        """Convert to Neo4j node properties."""
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "type": self.type,
            "vendor": self.vendor or "",
            "location": self.location or "",
            "active": self.active,
            "updated_at": (self.updated_at or datetime.utcnow()).isoformat(),
        }


@dataclass
class Status:
    """Status node — represents a machine state."""
    code: StatusCode
    severity: Optional[StatusSeverity] = None
    label: Optional[str] = None  # User-friendly label ("Running", "Tool change", etc.)
    
    def to_neo4j_properties(self) -> Dict[str, any]:
        """Convert to Neo4j node properties."""
        return {
            "code": self.code.value,
            "severity": self.severity.value if self.severity else "",
            "label": self.label or self.code.value,
        }


@dataclass
class Description:
    """Description node — metadata/documentation about a machine."""
    text: str                      # Main description (e.g., "5-axis milling center used for roughing and finishing.")
    lang: Optional[str] = "en"    # Language (ISO 639-1)
    source: Optional[str] = None  # Origin (e.g., "manual", "datasheet", "sindit")
    
    def to_neo4j_properties(self) -> Dict[str, any]:
        """Convert to Neo4j node properties."""
        return {
            "text": self.text,
            "lang": self.lang or "en",
            "source": self.source or "unknown",
        }


@dataclass
class StatusEvent:
    """StatusEvent node — history of status transitions."""
    timestamp: datetime
    from_status: StatusCode
    to_status: StatusCode
    reason: Optional[str] = None         # Why the transition occurred
    source: Optional[str] = None         # Source of the event (e.g., "sindit", "operator", "model")
    
    def to_neo4j_properties(self) -> Dict[str, any]:
        """Convert to Neo4j node properties."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "reason": self.reason or "",
            "source": self.source or "unknown",
        }


@dataclass
class ProductionLine:
    """ProductionLine node — groups related machines."""
    line_id: str                        # Unique identifier (e.g., "LINE-A")
    name: str                           # Display name
    location: Optional[str] = None
    
    def to_neo4j_properties(self) -> Dict[str, any]:
        """Convert to Neo4j node properties."""
        return {
            "line_id": self.line_id,
            "name": self.name,
            "location": self.location or "",
        }


# ============================================================================
# Neo4j Cypher schema definition
# ============================================================================

CYPHER_SCHEMA_DDL = """
// Create constraints and indexes for machine-centric schema

// Uniqueness constraints (prevents duplicate nodes)
CREATE CONSTRAINT machine_id IF NOT EXISTS FOR (m:Machine) REQUIRE m.machine_id IS UNIQUE;
CREATE CONSTRAINT status_code IF NOT EXISTS FOR (s:Status) REQUIRE s.code IS UNIQUE;
CREATE CONSTRAINT production_line_id IF NOT EXISTS FOR (p:ProductionLine) REQUIRE p.line_id IS UNIQUE;

// Indexes for query performance
CREATE INDEX machine_active IF NOT EXISTS FOR (m:Machine) ON (m.active);
CREATE INDEX machine_updated IF NOT EXISTS FOR (m:Machine) ON (m.updated_at);
CREATE INDEX status_event_timestamp IF NOT EXISTS FOR (e:StatusEvent) ON (e.timestamp);
CREATE INDEX status_event_source IF NOT EXISTS FOR (e:StatusEvent) ON (e.source);
"""

# Example seed data (Cypher)
CYPHER_SEED_DATA = """
// Create standard status nodes
CREATE (s_active:Status {code: "ACTIVE", severity: "INFO", label: "Running"})
CREATE (s_idle:Status {code: "IDLE", severity: "INFO", label: "Idle"})
CREATE (s_stopped:Status {code: "STOPPED", severity: "WARNING", label: "Stopped"})
CREATE (s_maintenance:Status {code: "MAINTENANCE", severity: "INFO", label: "Maintenance"})
CREATE (s_fault:Status {code: "FAULT", severity: "CRITICAL", label: "Fault"});

// Create example production line
CREATE (line:ProductionLine {line_id: "LINE-A", name: "Finishing Line", location: "Building 1"});

// Create example machines
CREATE (m1:Machine {
  machine_id: "MACHINE_A1",
  name: "5-axis Milling Center",
  type: "5-axis milling center",
  vendor: "DMG MORI",
  location: "Building 1, Station A",
  active: TRUE,
  updated_at: "2026-04-23T10:00:00Z"
});

CREATE (m2:Machine {
  machine_id: "MACHINE_B3",
  name: "Moving-Column Milling Center",
  type: "moving-column milling center",
  vendor: "KRONES",
  location: "Building 1, Station B",
  active: TRUE,
  updated_at: "2026-04-23T09:45:00Z"
});

// Link machines to production line
MATCH (line:ProductionLine {line_id: "LINE-A"}), (m1:Machine {machine_id: "MACHINE_A1"}), (m2:Machine {machine_id: "MACHINE_B3"})
CREATE (line)-[:HAS_MACHINE]->(m1)
CREATE (line)-[:HAS_MACHINE]->(m2);

// Set initial machine statuses
MATCH (m1:Machine {machine_id: "MACHINE_A1"}), (s:Status {code: "ACTIVE"})
CREATE (m1)-[:CURRENT_STATUS]->(s);

MATCH (m2:Machine {machine_id: "MACHINE_B3"}), (s:Status {code: "IDLE"})
CREATE (m2)-[:CURRENT_STATUS]->(s);

// Add descriptions
CREATE (d1:Description {
  text: "5-axis milling center used for roughing and finishing. Equipped with ATC for up to 50 tools.",
  lang: "en",
  source: "manual"
});

CREATE (d2:Description {
  text: "Moving-column milling center configured for large workpieces. Spindle speed up to 3000 RPM.",
  lang: "en",
  source: "datasheet"
});

MATCH (m1:Machine {machine_id: "MACHINE_A1"}), (d1:Description)
CREATE (m1)-[:HAS_DESCRIPTION]->(d1);

MATCH (m2:Machine {machine_id: "MACHINE_B3"}), (d2:Description)
CREATE (m2)-[:HAS_DESCRIPTION]->(d2);

// Add sample status events (history)
CREATE (e1:StatusEvent {
  timestamp: "2026-04-23T09:00:00Z",
  from_status: "STOPPED",
  to_status: "ACTIVE",
  reason: "Operator initiated start",
  source: "operator"
});

CREATE (e2:StatusEvent {
  timestamp: "2026-04-23T09:15:00Z",
  from_status: "ACTIVE",
  to_status: "IDLE",
  reason: "Job completed",
  source: "operator"
});

MATCH (m1:Machine {machine_id: "MACHINE_A1"}), (e1:StatusEvent)
CREATE (m1)-[:HAS_STATUS_EVENT]->(e1);

MATCH (m1:Machine {machine_id: "MACHINE_A1"}), (e2:StatusEvent)
CREATE (m1)-[:HAS_STATUS_EVENT]->(e2);
"""

# Query templates for common operations
QUERY_TEMPLATES = {
    "list_active_machines": """
        MATCH (m:Machine {active: TRUE})
        OPTIONAL MATCH (m)-[:CURRENT_STATUS]->(s:Status)
        OPTIONAL MATCH (m)-[:HAS_DESCRIPTION]->(d:Description)
        RETURN m.machine_id, m.name, s.code, s.label, d.text
        ORDER BY m.machine_id
    """,
    
    "get_machine_with_status": """
        MATCH (m:Machine {machine_id: $machine_id})
        OPTIONAL MATCH (m)-[:CURRENT_STATUS]->(s:Status)
        OPTIONAL MATCH (m)-[:HAS_DESCRIPTION]->(d:Description)
        RETURN m, s, d
    """,
    
    "get_machine_status_history": """
        MATCH (m:Machine {machine_id: $machine_id})-[:HAS_STATUS_EVENT]->(e:StatusEvent)
        RETURN e.timestamp, e.from_status, e.to_status, e.reason, e.source
        ORDER BY e.timestamp DESC
        LIMIT $limit
    """,
    
    "update_machine_status": """
        MATCH (m:Machine {machine_id: $machine_id})
        OPTIONAL MATCH (m)-[r:CURRENT_STATUS]->()
        DELETE r
        WITH m
        MATCH (s:Status {code: $status_code})
        CREATE (m)-[:CURRENT_STATUS]->(s)
        SET m.updated_at = $updated_at
        RETURN m, s
    """,
    
    "record_status_event": """
        MATCH (m:Machine {machine_id: $machine_id})
        CREATE (e:StatusEvent {
          timestamp: $timestamp,
          from_status: $from_status,
          to_status: $to_status,
          reason: $reason,
          source: $source
        })
        CREATE (m)-[:HAS_STATUS_EVENT]->(e)
        RETURN e
    """,
    
    "get_production_line": """
        MATCH (line:ProductionLine {line_id: $line_id})-[:HAS_MACHINE]->(m:Machine)
        OPTIONAL MATCH (m)-[:CURRENT_STATUS]->(s:Status)
        RETURN line, m, s
    """,
}
